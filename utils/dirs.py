"""
# Author = ruben
# Date: 27/8/23
# Project: retinal-framework
# File: config.py

Description: Module to load experiment configuration
"""
import os
import logging


def create_dirs(dirs: list):
    """list of directories to create if these directories are not found"""
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)
