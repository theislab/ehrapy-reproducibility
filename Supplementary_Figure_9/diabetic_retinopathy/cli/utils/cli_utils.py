import shutil
import sys
import traceback

import toml


def read_toml(config_path, cli_dict):
    print(f"Loading inference configurations from {config_path}:")
    config_dict = toml.load(config_path)
    for key in config_dict:
        value = config_dict[key]

        cli_dict[key] = value
    return cli_dict


def display_exception(ex, model_path=None):
    traceback.print_exc()

    if model_path is not None:
        shutil.rmtree(model_path)
