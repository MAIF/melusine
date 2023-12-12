"""
Default : Use default configuration to explore functionalities
Specify a config_dict
Specify a config path
Specify a MELUSINE_CONFIG_DIR environment variable
"""


def from_config():
    # --8<-- [start:from_config]
    from melusine.pipeline import MelusinePipeline

    pipeline = MelusinePipeline.from_config(config_key="demo_pipeline")


# --8<-- [end:from_config]


def from_config_dict():
    # --8<-- [start:from_config_dict]
    from melusine.processors import Normalizer

    normalizer_conf = {
        "input_columns": ["text"],
        "output_columns": ["normalized_text"],
        "form": "NFKD",
        "lowercase": False,
    }

    normalizer = Normalizer.from_config(config_dict=normalizer_conf)


# --8<-- [end:from_config_dict]


def print_config():
    # --8<-- [start:print_config]
    from melusine import config

    print(config["demo_pipeline"])


# --8<-- [end:print_config]


def modify_conf_with_dict():
    # --8<-- [start:modify_conf_with_dict]
    from melusine import config

    # Get a dict of the existing conf
    new_conf = config.dict()

    # Add/Modify a config key
    new_conf["my_conf_key"] = "my_conf_value"

    # Reset Melusine configurations
    config.reset(new_conf)


# --8<-- [end:modify_conf_with_dict]


def modify_conf_with_path():  # pragma: no cover
    """Tested in conf/test_config"""
    # --8<-- [start:modify_conf_with_path]
    from melusine import config

    # Specify the path to a conf folder
    conf_path = "path/to/conf/folder"

    # Reset Melusine configurations
    config.reset(config_path=conf_path)

    # >> Using config_path : path/to/conf/folder


# --8<-- [end:modify_conf_with_path]


def modify_conf_with_env():  # pragma: no cover
    """Tested in conf/test_config"""
    # --8<-- [start:modify_conf_with_env]
    import os

    from melusine import config

    # Specify the MELUSINE_CONFIG_DIR environment variable
    os.environ["MELUSINE_CONFIG_DIR"] = "path/to/conf/folder"

    # Reset Melusine configurations
    config.reset()

    # >> Using config_path from env variable MELUSINE_CONFIG_DIR
    # >> Using config_path : path/to/conf/folder


# --8<-- [end:modify_conf_with_env]


def export_config():  # pragma: no cover
    """Tested in conf/test_config"""
    # --8<-- [start:export_config]
    from melusine import config

    # Specify the path a folder (created if it doesn't exist)
    conf_path = "path/to/conf/folder"

    # Export default configurations to the folder
    files_created = config.export_default_config(path=conf_path)


# --8<-- [end:export_config]
