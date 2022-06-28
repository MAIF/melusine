import collections.abc
import json
import logging
import os
import os.path as op
import warnings
from pathlib import Path
from typing import Dict, Any

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


logger = logging.getLogger(__name__)


def update_nested_dict(base_dict: dict, update_dict: dict) -> dict:
    """
    Update a (possibly) nested dictionary using another (possibly) nested dictionary.
    Ex:
        base_dict = {"A": {"a": "0"}}
        u = {"A": {"b": "42"}}
        update_dict = update_nested_dict(d, u)
        # Output : {"A": {"a": "0", "b": "42"}}

    Parameters
    ----------
    base_dict: Mapping
        Base dict to be updated
    update_dict: Mapping
        Update dict to merge into d

    Returns
    -------
    base_dict: Mapping
        Updated dict
    """
    for key, value in update_dict.items():
        if isinstance(value, collections.abc.Mapping):
            base_dict[key] = update_nested_dict(base_dict.get(key, {}), value)
        else:
            base_dict[key] = value
    return base_dict


def load_conf_from_path(config_dir_path: str) -> Dict[str, Any]:
    """
    Given a directory path
    Parameters
    ----------
    config_dir_path: str
        Path to a directory containing YML or JSON conf files
    Returns
    -------
    conf: dict
        Loaded config dict
    """
    conf = dict()
    conf_files = list()
    conf_files.extend([str(f) for f in Path(config_dir_path).rglob("*.yml")])
    conf_files.extend([str(f) for f in Path(config_dir_path).rglob("*.yaml")])
    conf_files.extend([str(f) for f in Path(config_dir_path).rglob("*.json")])

    # Prevent loading notebook checkpoints
    conf_files = [x for x in conf_files if "ipynb_checkpoints" not in x]

    for name in conf_files:
        # Load YAML files
        if name.endswith(".yml") or name.endswith(".yaml"):
            logger.info(f"Loading data from file {name}")
            with open(name, "r") as f:
                tmp_conf = yaml.load(f, Loader=Loader)
                conf = update_nested_dict(conf, tmp_conf)

        # Load JSON files
        elif name.endswith(".json"):
            logger.info(f"Loading data from file {name}")
            with open(file=name, mode="r", encoding="utf-8") as f:
                tmp_conf = json.load(f)
                conf = update_nested_dict(conf, tmp_conf)

    return conf


class MelusineConfig:
    """
    The MelusineConfig class acts as a dict containing configurations.
    The configurations can be changed dynamically using the switch_config function.
    """

    def __init__(self):
        super().__init__()
        self._config = None
        self.load_melusine_conf()

    def __getitem__(self, key):
        """
        Access configuration elements
        """
        return self._config[key]

    def __repr__(self):
        """
        Represent the MelusineConfig class
        """
        return repr(self._config)

    def __len__(self):
        """
        Returns the length of the config dict
        """
        return len(self._config)

    def copy(self):
        """
        Copy the config dict
        """
        return self._config.copy()

    def has_key(self, k):
        """
        Checks if given key exists in the config dict
        """
        return k in self._config

    def keys(self):
        """
        Returns the keys of the config dict
        """
        return self._config.keys()

    def values(self):
        """
        Returns the values of the config dict
        """
        return self._config.values()

    def items(self):
        """
        Returns the items of the config dict
        """
        return self._config.items()

    def __contains__(self, item):
        """
        Checks if the given item is contained in the config dict
        """
        return item in self._config

    def __iter__(self):
        """
        Iterates over the the config dict
        """
        return iter(self._config)

    def load_melusine_conf(self) -> None:
        """
        Load the melusine configurations.
        The default configurations are loaded first (the one present in the melusine package).
        Custom configurations may overwrite the default ones.
        Custom configuration should be specified in YML and JSON files and placed in a directory.
        The directory path should be set as the value of the MELUSINE_CONFIG_DIR environment variable.
        Returns
        -------
        conf: dict
            Loaded config dict
        """
        conf = dict()

        # Load default Melusine conf
        default_config_directory = op.dirname(op.abspath(__file__))
        conf = update_nested_dict(conf, load_conf_from_path(default_config_directory))

        # Load custom Melusine conf
        custom_config_directory = os.getenv("MELUSINE_CONFIG_DIR")
        if custom_config_directory:
            conf = update_nested_dict(
                conf, load_conf_from_path(custom_config_directory)
            )

        self._switch_config(conf)

    def _switch_config(self, new_config):
        """
        Modify the private attribute _config of the MelusineConfig instance.

        Parameters
        ----------
        new_config: dict
        Dict containing the new config
        """
        config_deprecation_warnings(new_config)
        self._config = new_config
        logger.info(f"Updated config")


def switch_config(new_config):
    """
    Function to change the Melusine configuration using a dict.

    Parameters
    ----------
    new_config: dict
        Dict containing the new config
    """
    global config

    config._switch_config(new_config)


def config_deprecation_warnings(config_dict: Dict[str, Any]) -> None:
    """
    Raise Deprecation Warning when using deprecated configs
    """

    words_list = config_dict.get("words_list")
    if isinstance(words_list, dict) and words_list.get("stopwords"):
        logger.warning(
            "DeprecationWarning:"
            "Config words_list.stopwords is deprecated, please use tokenizer.stopwords"
        )
        warnings.warn(
            "Config words_list.stopwords is deprecated, please use tokenizer.stopwords",
            DeprecationWarning,
        )

    if isinstance(words_list, dict) and words_list.get("names"):
        logger.warning(
            "DeprecationWarning:"
            "Config words_list.names is deprecated, please use token_flagger.token_flags.flag_name"
        )
        warnings.warn(
            "Config words_list.names is deprecated, please use token_flagger.token_flags.flag_name",
            DeprecationWarning,
        )

    regex = config_dict.get("regex")
    if isinstance(regex, dict) and regex.get("tokenizer"):
        logger.warning(
            "DeprecationWarning:"
            "Config regex.tokenizer is deprecated, please use tokenizer.tokenizer_regex"
        )
        warnings.warn(
            "Config regex.tokenizer is deprecated, please use tokenizer.tokenizer_regex",
            DeprecationWarning,
        )

    if isinstance(regex, dict):
        cleaning = regex.get("cleaning")
        if isinstance(cleaning, dict) and cleaning.get("flags_dict"):
            logger.warning(
                "DeprecationWarning:"
                "Config regex.cleaning.flags_dict is deprecated, please use text_flagger.text_flags"
            )
            warnings.warn(
                "Config regex.cleaning.flags_dict is deprecated, please use text_flagger.text_flags",
                DeprecationWarning,
            )


# Load Melusine configurations
config = MelusineConfig()
