# -*- coding: utf-8 -*-
#
# main.py
#

"""
Features selection and classifications
"""

__author__ = "Ngoc Huynh,Ahmed Albuni"
__email__ = "ngoc.huynh.bao@nmbu.no,ahmed.albuni@gmail.com"

import argparse
import json

import imskaper.experiment.experiment as ex
import imskaper.validations.validate_config as validate

def imskaper_feature_selection():

    parser = argparse.ArgumentParser(
        description="Features selection and " "classifications (2 classes)"
    )
    parser.add_argument(
        "-file",
        type=str,
        help="Features JSON file name and "
        "path, response var y should be "
        "the last field",
    )

    args = parser.parse_args()

    with open(args.file) as config_file:
        config = json.load(config_file)
    if validate.validate_config_file(config):
        ex.experiment(config)


if __name__ == "__main__":
    imskaper_feature_selection()
