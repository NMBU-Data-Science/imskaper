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
import os.path


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
with open(path) as config_file:
    schema = json.load(config_file)


def validate_config_file(config):
    validate(config, schema=schema)
    try:
        validate(config, schema=schema)
    except ValidationError as err:
        print(err)
        return False
    return True
