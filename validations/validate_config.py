# -*- coding: utf-8 -*-
#
# validate_config.py
#

"""
Validate the configuration json file
"""

__author__ = "Ahmed Albuni"
__email__ = "ahmed.albuni@gmail.com"


from jsonschema import validate
from jsonschema.exceptions import ValidationError
import json

with open("validations\\schema.json") as config_file:
    schema = json.load(config_file)


def validate_config_file(config):
    try:
        validate(config, schema=schema)
    except ValidationError as err:
        print(err)
        return False
    return True
