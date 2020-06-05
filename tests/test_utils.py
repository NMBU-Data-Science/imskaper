import json
import os

import pytest

from utils.classifiers import get_classifiers
from utils.features_selectors import get_features_selectors

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_config.json"
)
with open(path) as config_file:
    config = json.load(config_file)


def test_get_classifiers():
    c_list = get_classifiers(config)
    assert isinstance(c_list, dict)
    assert len(c_list) > 0


def test_get_features_selectors():
    c_list = get_features_selectors(config)
    assert isinstance(c_list, dict)
    assert len(c_list) > 0
