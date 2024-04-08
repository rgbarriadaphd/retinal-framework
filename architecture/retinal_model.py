"""
# Author = ruben
# Date: 29/8/23
# Project: retinal-framework
# File: retinal_model.py

Description: Retinal model implementation by VGG16
"""

import torch.nn as nn
from torchvision import models


class RetinalModel(nn.Module):
    def __init__(self, num_classes: int, mode: str = 'basic_a'):
        """
        Implements VGG16 for classification with retinal images

        Args:
        num_classe: Number of classification categories
        mode: Defines the parts of the architecture that will be trained:
            ``'basic_a'`` (default): Pretrained with IMAGENET1K_V1: nothing freeze
            ``'basic_b'``: Pretrained with IMAGENET1K_V1: freeze features
            ``'features_a'``: Pretrained with IMAGENET1K_FEATURES: nothing freeze, classifier init with Xavier.
            ``'features_b'``: Pretrained with IMAGENET1K_FEATURES: freeze features, classifier init with Xavier.
        """
        super(RetinalModel, self).__init__()
        self._model = None
        self._num_classes = num_classes
        self._mode = mode
        self._last_fc_index = 6

    def _init_model(self):
        if self._mode == 'basic_a' or self._mode == 'basic_b':
            self._model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            self._model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)

        num_features = self._model.classifier[self._last_fc_index].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, self._num_classes)
        features.extend([linear])
        self._model.classifier = nn.Sequential(*features)

    def _freeze_sub_architecture(self):
        if self._mode == 'basic_a' or self._mode == 'features_a':
            for param in self._model.parameters():
                param.requires_grad = True
        elif self._mode == 'basic_b' or self._mode == 'features_b':
            for param in self._model.features.parameters():
                param.requires_grad = False
            for param in self._model.classifier.parameters():
                param.requires_grad = True

    def _weight_initialization(self):
        if self._mode == 'features_a' or self._mode == 'features_b':
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)

    def get(self):
        """
        Return model
        """
        self._init_model()
        self._freeze_sub_architecture()
        self._weight_initialization()
        return self._model
