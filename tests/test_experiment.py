import json
import os
import shutil

from experiment.experiment import experiment, read_Xy_data
from utils.classifiers import get_classifiers
from utils.features_selectors import get_features_selectors

current_path = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_path, "test_config.json")
with open(json_path) as config_file:
    config = json.load(config_file)
config["config"]["features_file"] = os.path.join(
    current_path, "test_dataset.csv"
)
config["config"]["output_dir"] = os.path.join(current_path, "temp\\")


def test_read_Xy():
    X, y, columns_names = read_Xy_data(config["config"]["features_file"])
    assert X.shape == (197, 22)
    assert y.shape == (197,)
    assert len(columns_names) == 22
    assert columns_names[0] == "Elongation"


def test_exp():
    p = config["config"]["output_dir"]
    folder_to_delete = [os.path.join(p, f) for f in os.listdir(p)]
    for f in folder_to_delete:
        shutil.rmtree(f)
    df = experiment(config, verbose=0)
    assert df.shape == (
        len(get_classifiers(config)),
        len(get_features_selectors(config)),
    )
    folder_name = [os.path.join(p, f) for f in os.listdir(p)]
    assert len([f for f in os.listdir(folder_name[0])]) == 8
    df2 = experiment(config, verbose=0)
    # Make sure we are able to reproduce the results when using the same seed
    assert df.equals(df2)
    config["config"]["SEED"] = 999
    df3 = experiment(config, verbose=0)
    assert not (df.equals(df3))
    folder_to_delete = [os.path.join(p, f) for f in os.listdir(p)]
    for f in folder_to_delete:
        shutil.rmtree(f)
