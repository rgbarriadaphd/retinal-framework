"""
# Author = ruben
# Date: 27/8/23
# Project: retinal-framework
# File: retinal_main.py

Description: Main entry script point
"""

import argparse
from utils.config import *

from agents import *


def main():
    # Parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    mydict = config.general.agent
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.general.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
