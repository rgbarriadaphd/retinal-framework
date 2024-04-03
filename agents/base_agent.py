"""
# Author = ruben
# Date: 27/8/23
# Project: retinal-framework
# File: base_agent.py

Description: Base Agent class, where all other agents inherit from, that contains definitions
for all the necessary functions
"""

import logging


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config: dict):
        self._config = config
        self._logger = logging.getLogger("Agent")
        self._logger.info(" THE Configuration of your experiment ..")
        self._logger.info(self._config)

    def load_checkpoint(self, file_name: str):
        """Latest checkpoint loader"""
        raise NotImplementedError

    def save_checkpoint(self, file_name: str):
        """Checkpoint saver"""
        raise NotImplementedError

    def run(self):
        """The main operator"""
        raise NotImplementedError

    def train(self):
        """Main training loop
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """One epoch of training"""
        raise NotImplementedError

    def validate(self):
        """One cycle of model validation
        """
        raise NotImplementedError

    def finalize(self):
        """Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader"""
        raise NotImplementedError
