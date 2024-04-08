"""
# Author = ruben
# Date: 30/8/23
# Project: retinal-framework
# File: cardio_retinal_dataset.py

Description: "Enter description here"
"""
import glob
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from utils.config import get_config_from_json


class LoadRetinalData:
    """Support class that loads image data location"""

    def __init__(self, image_folder: str, fold_config_file: str, data_classes: dict):
        """Constructor class"""
        # Where images are located
        self._image_folder = image_folder

        # json configuration file with image fold distribution
        self._fold_config_file = fold_config_file

        # Data classes. Corresponding to NO/YES (negative cardio event and positive)
        self._data_classes = data_classes

        # Load json data
        self._load_fold_configuration()

    def _load_fold_configuration(self):
        """Reads json fold file into a dict"""
        self._config = get_config_from_json(self._fold_config_file)[0]

    def _select_image_files(self, outer: str, inner: str) -> (dict, dict):
        """
        Get list of images of train/test dataset order by class
        :param outer: outer fold
        :param inner: inner fold

        Note: Given fold (outer, inner) from the distribution represent the one used in test, the rest are just
        used in train distribution.
        Example:
            outer, inner: (5, 3)

            train folds: (5, 1),(5, 2),(5, 4),(5, 5)
            test folds: (5, 3)
        """
        # select images for the test set
        test_selection = self._config[outer][inner]

        # select images for the train set
        train_folds = [e for e in ['1', '2', '3', '4', '5'] if e != inner]
        ce_yes = []
        ce_no = []
        for tf in train_folds:
            ce_yes.extend(self._config[outer][tf]['YES'])
            ce_no.extend(self._config[outer][tf]['NO'])
        train_selection = {'NO': ce_no, 'YES': ce_yes}

        return train_selection, test_selection

    def _get_image_paths(self, train_selection: dict, test_selection: dict) -> (list, list):
        """
        retrieve image paths for both eyes, sets and fold
        :param train_selection: train images
        :param test_selection: test images
        """
        # Load image files data
        test_files = []
        train_files = []
        for cls in ['NO', 'YES']:
            for image_id in test_selection[cls]:
                pattern = os.path.join(self._image_folder, cls, image_id)
                for f in glob.glob(pattern):
                    test_files.append(f)

            for image_id in train_selection[cls]:
                pattern = os.path.join(self._image_folder, cls, image_id)
                for f in glob.glob(pattern):
                    train_files.append(f)

        # dynamic check of length
        assert sum([len(test_selection[k]) for k in test_selection]) == len(test_files)
        assert sum([len(train_selection[k]) for k in train_selection]) == len(train_files)

        return train_files, test_files

    def _get_sample_list(self, train_files: list, test_files: list) -> (list, list):
        """
        Associate samples with its corresponding label
        :param train_files: train images
        :param test_files: test images
        """
        train_samples = []
        test_samples = []
        # read data
        for test_image in test_files:
            key = 'NO' if '/NO' in test_image else 'YES'
            label = self._data_classes[key]
            test_samples.append((test_image, label))

        for train_image in train_files:
            key = 'NO' if '/NO' in train_image else 'YES'
            label = self._data_classes[key]
            train_samples.append((train_image, label))
        return train_samples, test_samples

    def create_fold_dataset(self, outer: str, inner: str) -> (list, list):
        """
        Reads data location and creates train and test data list from the corresponding fold
        :param outer: outer fold
        :param inner: inner fold
        """
        # select image files for both sets and corresponding fold
        train_selection, test_selection = self._select_image_files(outer, inner)

        # retrieve image paths for both eyes, sets and fold
        train_files, test_files = self._get_image_paths(train_selection, test_selection)

        # get samples with label association
        train_samples, test_samples = self._get_sample_list(train_files, test_files)

        return train_samples, test_samples


class CardioRetinalImageDataset(Dataset):
    """Class to iterate batches from cardio images"""

    def __init__(self, data: list, data_transforms: transforms):
        """Class constructor"""
        # image list
        self._retinal_data = data

        # transformation to be applied at get item
        self._transforms = data_transforms

    def __len__(self) -> int:
        return len(self._retinal_data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        #  extracting image from index and scaling
        image_path = self._retinal_data[idx][0]
        with open(image_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        # get label
        label = torch.tensor(self._retinal_data[idx][1])

        # apply transformations
        if self._transforms:
            image = self._transforms(image)

        return image, label
