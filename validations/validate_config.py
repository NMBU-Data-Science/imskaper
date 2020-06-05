# -*- coding: utf-8 -*-
#
# validate_config.py
#

"""
Validate the configuration json file
"""

__author__ = "Ahmed Albuni"
__email__ = "ahmed.albuni@gmail.com"


import json
import os.path

from jsonschema import validate
from jsonschema.exceptions import ValidationError

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
with open(path) as config_file:
    schema = json.load(config_file)
score_fun_list = (
    "f1",
    "accuracy",
    "roc_auc",
    "precision",
    "f1_micro",
    "f1_macro",
    "f1_weighted",
    "precision_micro",
    "precision_macro",
    "precision_weighted",
    "recall",
    "recall_micro",
    "recall_macro",
    "recall_weighted",
)


def validate_config_file(config):
    validate(config, schema=schema)
    try:
        validate(config, schema=schema)
    except ValidationError as err:
        print(err)
        return False
    score_fun = config["config"]["SCORE_FUN"]
    if score_fun not in score_fun_list:
        print(
            "Score function: ",
            score_fun,
            "is not allowed\n",
            "Allowed " "score " "functions" ": ",
            score_fun_list,
        )
        return False
    return True
