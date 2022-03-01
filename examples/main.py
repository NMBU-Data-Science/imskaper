# -*- coding: utf-8 -*-
#
# main.py
#

"""
Features selection and classifications
"""

__author__ = "Ahmed Albuni"
__email__ = "ahmed.albuni@gmail.com"


import argparse
import json

import imskaper.experiment.experiment as ex
import imskaper.validations.validate_config as validate

parser = argparse.ArgumentParser(
    description="Features selection and " "classifications (2 classes)"
)
parser.add_argument(
    "-file",
    type=str,
    help="Features CSV file name and "
    "path, response var y should be "
    "the last field",
)

if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.file) as config_file:
        config = json.load(config_file)
    if validate.validate_config_file(config):
        ex.experiment(config)
