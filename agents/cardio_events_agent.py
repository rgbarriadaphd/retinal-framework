"""
# Author = ruben
# Date: 27/8/23
# Project: retinal-framework
# File: cardio_events_agent.py

Description: Cardio events agent
"""
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.optim as optim
from tensorboardX import SummaryWriter
from torchvision import transforms

from agents.base_agent import BaseAgent
# from architecture.cac_retinal_model import CACRetinalModel
#
# from torch.optim.lr_scheduler import StepLR
#
# from dataset.cardio_retinal_dataset import LoadRetinalData
# from dataset.cardio_retinal_dataset import CardioRetinalImageDataset
#
# from utils.metrics import CrossValidationMeasures, PerformanceMetrics


class CardioEventsAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # get config
        self._config = config

        # define device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        torch.manual_seed(self._config.execution.seed)

        # Init folds
        self._outer = None
        self._inner = None

        # metrics
        self._metrics = {'f_accuracies': [], 'f_performance': {}}

        # Tensorboard Writer
        self._summary_writer = SummaryWriter(log_dir=self._config.summary_dir, comment='CACModel')

    def _start_process(self):
        """
        :return:
        """
        image_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', self._config.paths.clinical_dataset))
        fold_distribution_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', self._config.paths.fold_configuration))
        data_classes = self._config.general.classes
        optimizer = self._config.general.optimizer
        mean = self._config.transformations.image_normalization[0]
        std = self._config.transformations.image_normalization[1]
        folds = self._config.execution.folds

        # load_retinal_data = LoadRetinalData(image_folder, fold_distribution_file, data_classes)
        # data_transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std),
        # ])
        #
        # for outer in range(folds[0][0], folds[0][1] + 1):
        #     self._metrics['f_performance'][outer] = {}
        #     for inner in range(folds[1][0], folds[1][1] + 1):
        #         # initialize counter
        #         self._current_epoch = 1
        #         self._outer = outer
        #         self._inner = inner
        #
        #         # define model
        #         self._model = CACRetinalModel(num_classes=len(self._config.general.classes)).get()
        #         self._model = self._model.to(self._device)
        #
        #         # define loss
        #         self._criterion = nn.CrossEntropyLoss()
        #         self._criterion = self._criterion.to(self._device)
        #
        #         # define optimizer
        #         if optimizer == "SDG":
        #             self._optimizer = optim.SGD(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "Adadelta":
        #             self._optimizer = optim.Adadelta(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "Adagrad":
        #             self._optimizer = optim.Adagrad(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "Adam":
        #             self._optimizer = optim.Adam(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "AdamW":
        #             self._optimizer = optim.AdamW(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "Adamax":
        #             self._optimizer = optim.Adamax(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "ASGD":
        #             self._optimizer = optim.ASGD(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "LBFGS":
        #             self._optimizer = optim.LBFGS(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "NAdam":
        #             self._optimizer = optim.NAdam(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "RAdam":
        #             self._optimizer = optim.RAdam(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "RMSprop":
        #             self._optimizer = optim.RMSprop(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #         elif optimizer == "Rprop":
        #             self._optimizer = optim.Rprop(self._model.parameters(), lr=self._config.hyperparams.learning_rate)
        #
        #
        #
        #         self._scheduler = StepLR(self._optimizer, step_size=1, gamma=self._config.hyperparams.gamma,
        #                                  verbose=True)
        #
        #         # Get fold data
        #         train_fold_data, test_fold_data = load_retinal_data.create_fold_dataset(str(outer), str(inner))
        #
        #         # load train dataset
        #         train_data_set = CardioRetinalImageDataset(data=train_fold_data, transforms=data_transforms)
        #         train_dl = DataLoader(train_data_set, batch_size=self._config.hyperparams.batch_size, shuffle=True, num_workers=4,
        #                               pin_memory=True)
        #
        #         # load test dataset
        #         test_data_set = CardioRetinalImageDataset(data=test_fold_data, transforms=data_transforms)
        #         test_dl = DataLoader(test_data_set, batch_size=1, shuffle=True, num_workers=4,
        #                               pin_memory=True)
        #
        #         self._train(train_dl, test_dl)
        #         accuracy, performance = self._test(test_dl)
        #         self._metrics['f_accuracies'].append(accuracy)
        #         self._metrics['f_performance'][outer][inner] = performance
        #         self._logger.info("\n*****\nModel Accuracy: {}\n*****\n".format(accuracy))
        #
        # for outer in range(folds[0][0], folds[0][1] + 1):
        #     for inner in range(folds[1][0], folds[1][1] + 1):
        #         self._logger.info("[{},{}]: {}".format(outer, inner, self._metrics['f_performance'][outer][inner]))
        #
        # cvm = CrossValidationMeasures(measures_list=self._metrics['f_accuracies'], percent=True, formatted=True)
        # self._logger.info(f'Fold results: {self._metrics["f_accuracies"]}')
        # self._logger.info(f'Mean: {cvm.mean()}')
        # self._logger.info(f'Std. Dev: {cvm.stddev()}')
        # self._logger.info(f'C.Interval: {cvm.interval()}')

    def _train(self, train_dl, test_dl):
        for epoch in range(1, self._config.hyperparams.epochs + 1):
            self._train_epoch(train_dl, test_dl)
            # self._scheduler.step()
            self._current_epoch += 1

    def _train_epoch(self, train_loader, test_loader):
        """
        Main training loop
        :return:
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

        if self._current_epoch % self._config.hyperparams.validate_every == 0:
            acc_train = self._evaluate_process(train_loader, 'train')
            acc_test = self._evaluate_process(test_loader, 'test')
            self._summary_writer.add_scalars(f'epoch/accuracy_{self._outer}_{self._inner}', {
                'train': acc_train,
                'test': acc_test,
            }, self._current_epoch)

        self._summary_writer.add_scalar(f'epoch/loss_{self._outer}_{self._inner}', epoch_loss, self._current_epoch)

    def _test(self, dataloader):
        """
        :return:
        """
        self._model.eval()
        # correct = 0
        # ground_array = []
        # prediction_array = []
        # with torch.no_grad():
        #     for data, target in dataloader:
        #         data, target = data.to(self._device), target.to(self._device)
        #         output = self._model(data)
        #         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #         correct += pred.eq(target.view_as(pred)).sum().item()
        #
        #         ground_array.append(target.item())
        #         prediction_array.append(pred.item())
        #
        # self._logger.info(f'\n-----------\nAccuracy test set: {100. * correct / len(dataloader.dataset):.2f}')
        # pm = PerformanceMetrics(ground=ground_array,
        #                         prediction=prediction_array,
        #                         percent=True,
        #                         formatted=True)
        # confusion_matrix = pm.confusion_matrix()
        # performance = {
        #     f'accuracy:': pm.accuracy(),
        #     f'precision:': pm.precision(),
        #     f'recall:': pm.recall(),
        #     f'f1:': pm.f1(),
        #     f'tn:': confusion_matrix[0],
        #     f'fp:': confusion_matrix[1],
        #     f'fn:': confusion_matrix[2],
        #     f'tp:': confusion_matrix[3]
        # }
        # return 100. * correct / len(dataloader.dataset), performance


    def _evaluate_process(self, dataloader, stage):
        """
        :return:
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

        self._logger.info(f'\t>> Accuracy over ({stage}) set : {100. * correct / len(dataloader.dataset):.2f}')
        return 100. * correct / len(dataloader.dataset)


    def run(self):
        """
        The main operator
        :return:
        """
        self._t0 = time.time()
        try:
            self._start_process()
        except KeyboardInterrupt:
            self._logger.info("You have entered CTRL+C.. Wait to finalize")


    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self._tt = time.time() - self._t0
        self._logger.info("--------------------------------------------------")
        self._logger.info("Elapsed time: {}".format(self._tt))
        self._logger.info("Please wait while finalizing the operation. Thank you")
        self._summary_writer.close()