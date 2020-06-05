import json
import os.path

import pytest
from jsonschema.exceptions import ValidationError

import validations.validate_config as validate

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_config.json"
)
with open(path) as config_file:
    config = json.load(config_file)


@pytest.fixture(autouse=True)
def reset_config_file():
    global config
    with open(path) as config_file:
        config = json.load(config_file)


def test_valid_config_file():
    assert validate.validate_config_file(config)


def test_invalid_cv():
    config["config"]["CV"] = 1
    with pytest.raises(ValidationError):
        validate.validate_config_file(config)
    config["config"]["CV"] = 0
    with pytest.raises(ValidationError):
        validate.validate_config_file(config)
    config["config"]["CV"] = -1
    with pytest.raises(ValidationError):
        validate.validate_config_file(config)
    config["config"]["CV"] = "s"
    with pytest.raises(ValidationError):
        validate.validate_config_file(config)


def test_seed():
    config["config"]["SEED"] = "S"
    with pytest.raises(ValidationError):
        validate.validate_config_file(config)
