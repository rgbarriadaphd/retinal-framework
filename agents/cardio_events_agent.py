"""
# Author = ruben
# Date: 27/8/23
# Project: retinal-framework
# File: cardio_events_agent.py

Description: Cardio events agent
"""
import os
import time
from easydict import EasyDict
import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms

from agents.base_agent import BaseAgent
from architecture.retinal_model import RetinalModel
from torch.optim.lr_scheduler import StepLR

from dataset.cardio_retinal_dataset import LoadRetinalData
from dataset.cardio_retinal_dataset import CardioRetinalImageDataset

from utils.metrics import CrossValidationMeasures, PerformanceMetrics


class CardioEventsAgent(BaseAgent):
    """
    Manages CE process
    """

    def __init__(self, config: EasyDict):
        """
        Agent constructor
        :param config: gathers all experiment configuration given by user
        """
        super().__init__(config)

        # set config
        self._config = config

        # Set device environment
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # set seed. Force deterministic behaviour in random proces for the shake of reproducibility.
        torch.manual_seed(self._config.execution.seed)

        # Init folds for cross validation results
        self._outer = None
        self._inner = None

        # timing variables
        self._tt = 0.0
        self._t0 = 0.0

        # output metrics structure
        self._metrics = {'f_accuracies': [], 'f_performance': {}}

        # Tensorboard Writer to generate plots.
        self._summary_writer = SummaryWriter(log_dir=self._config.summary_dir, comment='CEModel')

    def _start_process(self):
        """ Initialize learning procedure"""
        # Retrieve relevant parameters
        image_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', self._config.paths.clinical_dataset))
        fold_distribution_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', self._config.paths.fold_configuration))
        data_classes = self._config.general.classes
        mean = self._config.transformations.image_normalization[0]
        std = self._config.transformations.image_normalization[1]
        folds = self._config.execution.folds

        # Load data location and fold distribution.
        load_retinal_data = LoadRetinalData(image_folder, fold_distribution_file, data_classes)
        # Define data transformation that will be applied to image
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        for outer in range(folds[0][0], folds[0][1] + 1):
            self._metrics['f_performance'][outer] = {}
            for inner in range(folds[1][0], folds[1][1] + 1):
                # initialize epoch counter and folds ids
                self._current_epoch = 1
                self._outer = outer
                self._inner = inner

                # define model
                self._model = RetinalModel(num_classes=len(self._config.general.classes)).get()
                self._model = self._model.to(self._device)

                # define loss function
                self._criterion = nn.CrossEntropyLoss()
                self._criterion = self._criterion.to(self._device)

                # define optimizer
                self._optimizer = optim.SGD(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
                self._scheduler = StepLR(self._optimizer, step_size=1, gamma=self._config.hyperparams.gamma,
                                         verbose=True)

                # Get fold data
                train_fold_data, test_fold_data = load_retinal_data.create_fold_dataset(str(outer), str(inner))

                # load train dataset
                train_data_set = CardioRetinalImageDataset(data=train_fold_data, transforms=data_transforms)
                train_dl = DataLoader(train_data_set, batch_size=self._config.hyperparams.batch_size, shuffle=True,
                                      num_workers=4,
                                      pin_memory=True)

                # load test dataset
                test_data_set = CardioRetinalImageDataset(data=test_fold_data, transforms=data_transforms)
                test_dl = DataLoader(test_data_set, batch_size=1, shuffle=True, num_workers=4,
                                     pin_memory=True)

                # Train model on current fold distribution
                self._train(train_dl, test_dl)

                # Test model
                accuracy, performance = self._test(test_dl)
                self._metrics['f_accuracies'].append(accuracy)
                self._metrics['f_performance'][outer][inner] = performance
                self._logger.info("\n*****\nModel Accuracy: {}\n*****\n".format(accuracy))

        # Show data after computation
        for outer in range(folds[0][0], folds[0][1] + 1):
            for inner in range(folds[1][0], folds[1][1] + 1):
                self._logger.info("[{},{}]: {}".format(outer, inner, self._metrics['f_performance'][outer][inner]))

        # Cross-validation data
        cvm = CrossValidationMeasures(measures_list=self._metrics['f_accuracies'], percent=True, formatted=True)
        self._logger.info(f'Fold results: {self._metrics["f_accuracies"]}')
        self._logger.info(f'Mean: {cvm.mean()}')
        self._logger.info(f'Std. Dev: {cvm.stddev()}')
        self._logger.info(f'C.Interval: {cvm.interval()}')

    def _train(self, train_dl: DataLoader, test_dl: DataLoader):
        """
        Manages train model train process
        :param train_dl: local train dataloader
        :param test_dl: local test dataloader
        """
        for epoch in range(1, self._config.hyperparams.epochs + 1):
            # run one epoch
            self._train_epoch(train_dl, test_dl)

            # update learning rate if needed
            self._scheduler.step()
            self._current_epoch += 1

    def _train_epoch(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Train a single epoch.
        :param train_loader: fold train dataloader
        :param test_loader: fold test dataloader. test dataloader will be use just for validation plot
        for model debug purposes
        """
        self._model.train()
        n_train = len(train_loader.dataset)
        running_loss = 0.0
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self._device), target.to(self._device)
            self._optimizer.zero_grad()
            output = self._model(data)
            loss = self._criterion(output, target)
            loss.backward()
            self._optimizer.step()

            current_batch_size = data.size(0)
            running_loss += loss.item() * current_batch_size
            epoch_loss = running_loss / n_train

        self._logger.info(f'[{self._outer},{self._inner}] Train Epoch [{self._current_epoch}]: Loss: {loss.item():.4f}')

        # Plot train and test process
        if self._current_epoch % self._config.hyperparams.validate_every == 0:
            acc_train = self._evaluate_process(train_loader, 'train')
            acc_test = self._evaluate_process(test_loader, 'test')
            self._summary_writer.add_scalars(f'epoch/accuracy_{self._outer}_{self._inner}', {
                'train': acc_train,
                'test': acc_test,
            }, self._current_epoch)

        self._summary_writer.add_scalar(f'epoch/loss_{self._outer}_{self._inner}', epoch_loss, self._current_epoch)

    def _test(self, dataloader: DataLoader) -> (float, dict):
        """
        Test the model
        :param dataloader: test dataloader
        """
        self._model.eval()
        correct = 0
        ground_array = []
        prediction_array = []
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self._device), target.to(self._device)
                output = self._model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                # Save prediction/ground vectors to obtain prediction metrics
                ground_array.append(target.item())
                prediction_array.append(pred.item())

        self._logger.info(f'\n-----------\nAccuracy test set: {100. * correct / len(dataloader.dataset):.2f}')
        # Compute prediction metrics
        pm = PerformanceMetrics(ground=ground_array,
                                prediction=prediction_array,
                                percent=True,
                                formatted=True)
        confusion_matrix = pm.confusion_matrix()
        performance = {
            f'accuracy:': pm.accuracy(),
            f'precision:': pm.precision(),
            f'recall:': pm.recall(),
            f'f1:': pm.f1(),
            f'tn:': confusion_matrix[0],
            f'fp:': confusion_matrix[1],
            f'fn:': confusion_matrix[2],
            f'tp:': confusion_matrix[3]
        }
        return 100. * correct / len(dataloader.dataset), performance

    def _evaluate_process(self, dataloader: DataLoader, stage: str) -> float:
        """
        Evaluate given dataloader over model
        :param dataloader: test dataloader
        :param stage: 'train' | 'test'
        """
        n_test = len(dataloader.dataset)
        self._model.eval()

        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self._device), target.to(self._device)

                output = self._model(data)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        self._logger.info(f'\t>> Accuracy over ({stage}) set : {100. * correct / n_test:.2f}')
        return 100. * correct / n_test

    def run(self):
        """The main computation"""
        self._t0 = time.time()
        try:
            self._start_process()
        except KeyboardInterrupt:
            self._logger.info("You have entered CTRL+C.. Wait to finalize")

    def finalize(self):
        """Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader"""
        self._tt = time.time() - self._t0
        self._logger.info("--------------------------------------------------")
        self._logger.info("Elapsed time: {}".format(self._tt))
        self._logger.info("Please wait while finalizing the operation. Thank you")
        self._summary_writer.close()
