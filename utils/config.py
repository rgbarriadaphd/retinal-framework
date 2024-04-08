"""
# Author = ruben
# Date: 27/8/23
# Project: retinal-framework
# File: config.py

Description: Module to load experiment configuration
"""

import os

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
from easydict import EasyDict

from utils.dirs import create_dirs


def setup_logging(log_dir: str):
    """Init logger"""
    # log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_file_format = "[%(levelname)s]: %(message)s"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file: str) -> tuple:
    """Get the config from a json file"""

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(json_file: str) -> EasyDict:
    """ Initializes experiment"""
    # Loads experiment configuration
    config, _ = get_config_from_json(json_file)

    # making sure that you have provided the exp_name.
    try:
        print(" *************************************** ")
        print(" The experiment name is: {}".format(config.general.exp_name))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)

    # Create folders.
    summary_dir = os.path.join("experiments", config.general.exp_name, "summaries/")
    checkpoint_dir = os.path.join("experiments", config.general.exp_name, "checkpoints/")
    out_dir = os.path.join("experiments", config.general.exp_name, "out/")
    log_dir = os.path.join("experiments", config.general.exp_name, "logs/")
    create_dirs([summary_dir, checkpoint_dir, out_dir, log_dir])

    # setup logging in the project
    setup_logging(log_dir)

    logging.getLogger().info("Configurations are successfully processed and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")

    return config
