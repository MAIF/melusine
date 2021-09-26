import json
import os
import os.path as op
import logging
import glob
import yaml
import collections.abc

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


logger = logging.getLogger(__name__)


def update_nested_dict(d, u):
    """
    Update a (possibly) nested dictionary using another (possibly) nested dictionary.
    Ex:
        d = {"A": {"a": "0"}}
        u = {"A": {"b": "42"}}
        d = update_nested_dict(d, u)
        # Output : {"A": {"a": "0", "b": "42"}}
    Parameters
    ----------
    d: Mapping
        Base dict to be updated
    u: Mapping
        Update dict to merge into d
    Returns
    -------
    d: Mapping
        Updated dict
    """
    for key, value in u.items():
        if isinstance(value, collections.abc.Mapping):
            d[key] = update_nested_dict(d.get(key, {}), value)
        else:
            d[key] = value
    return d


def load_conf_from_path(config_dir_path):
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

    for name in glob.glob(f"{config_dir_path}/*"):

        # Load YAML files
        if name.endswith(".yml"):
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


def load_melusine_conf():
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
        conf = update_nested_dict(conf, load_conf_from_path(custom_config_directory))

    return conf


def config_retro_compatibility(config_):
    deprecation_message_template = """
        Deprecation warning : Melusine configurations have been updated
        Found config key config{old}
        This should be placed in config{new}    
    """
    if not config_.get("tokenizer"):
        config_["tokenizer"] = dict()

    config_regex = config_.get("regex")
    config_words_list = config_.get("words_list")
    if config_regex:
        if config_regex.get("tokenizer") and not config_["tokenizer"].get(
            "tokenizer_regex"
        ):
            config_["tokenizer"]["tokenizer_regex"] = config_regex["tokenizer"]
            logger.warning(
                deprecation_message_template.format(old="""["regex"]["tokenizer"]"""),
                new="""["tokenizer"]["tokenizer_regex"]""",
            )
        if config_regex.get("flag_dict") and not config_["tokenizer"].get("flag_dict"):
            config_["tokenizer"]["flag_dict"] = config_regex["flag_dict"]
            logger.warning(
                deprecation_message_template.format(old="""["regex"]["flag_dict"]"""),
                new="""["tokenizer"]["flag_dict"]""",
            )

    if config_words_list:
        if config_words_list.get("stopwords") and not config_["tokenizer"].get(
            "stopwords"
        ):
            config_["tokenizer"]["stopwords"] = config_words_list["stopwords"]
            logger.warning(
                deprecation_message_template.format(
                    old="""["words_list"]["stopwords"]"""
                ),
                new="""["tokenizer"]["stopwords"]""",
            )
        if config_words_list.get("names") and not config_["tokenizer"].get("names"):
            config_["tokenizer"]["names"] = config_words_list["names"]
            logger.warning(
                deprecation_message_template.format(old="""["words_list"]["names"]"""),
                new="""["tokenizer"]["names"]""",
            )

    return config_


# Load Melusine configurations
config = load_melusine_conf()
config = config_retro_compatibility(config)
