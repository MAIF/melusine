"""
Module which handles the package configuration.
"""
import copy
import logging
import os
from collections import UserDict
from pathlib import Path
from typing import Any, Dict, List, Optional, cast, no_type_check

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
CONST_ENV_MELUSINE_CONFIG_DIR = "MELUSINE_CONFIG_DIR"


class MelusineConfig(UserDict):
    """
    The MelusineConfig class acts as a dict containing configurations.
    The configurations can be changed dynamically using the switch_config function.
    """

    ENV_MELUSINE_CONFIG_DIR = "MELUSINE_CONFIG_DIR"
    LOG_MESSAGE_DEFAULT_CONFIG = "Using default configurations."
    LOG_MESSAGE_CONFIG_FROM_ENV_VARIABLE = f"Using config_path from env variable {ENV_MELUSINE_CONFIG_DIR}."
    LOG_MESSAGE_CONFIG_PATH = "Using config_path : {config_path}."
    DEFAULT_CONFIG_PATH = str(Path(__file__).parent.resolve() / "conf")

    @no_type_check
    def pop(self, s: Any = None) -> None:
        """
        Prevent MelusineConfig modification.
        """
        raise MelusineConfigError()

    @no_type_check
    def popitem(self, s: Any = None) -> None:
        """
        Prevent MelusineConfig modification.
        """
        raise MelusineConfigError()

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Prevent MelusineConfig modification.
        """
        raise MelusineConfigError()

    def dict(self) -> Dict[str, Any]:
        """
        Return a copy of the config dict.
        """
        return copy.deepcopy(self.data)

    @staticmethod
    def _load_from_path(config_path: str) -> Dict[str, Any]:
        """
        Load yaml config files, merge them and return a config dict.
        """
        yaml_conf_file_list = list(Path(config_path).rglob("*.yaml")) + list(Path(config_path).rglob("*.yml"))
        omega_conf = OmegaConf.unsafe_merge(*[OmegaConf.load(conf_file) for conf_file in yaml_conf_file_list])
        return cast(Dict[str, Any], OmegaConf.to_object(omega_conf))

    def reset(self, config_dict: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None) -> None:
        """
        Function to reset the Melusine configuration using a dict or a path.

        Parameters
        ----------
            config_dict: Dict containing the new config.
            config_path: path to directory containing YAML config files.
        """
        config_path_from_env = os.getenv(self.ENV_MELUSINE_CONFIG_DIR)

        if not config_dict and not config_path:
            if config_path_from_env:
                logger.info(self.LOG_MESSAGE_CONFIG_FROM_ENV_VARIABLE)
                config_path = config_path_from_env
            else:
                logger.warning(self.LOG_MESSAGE_DEFAULT_CONFIG)
                config_dict = self._load_from_path(self.DEFAULT_CONFIG_PATH)

        if config_path:
            logger.info(self.LOG_MESSAGE_CONFIG_PATH.format(config_path=config_path))
            config_dict = self._load_from_path(config_path)

        if config_dict is None:
            raise MelusineConfigError()  # pragma no cover

        self.data = config_dict

    def export_default_config(self, path: str) -> List[str]:
        """
        Export the default Melusine configurations to a directory.

        Parameters
        ----------
        path: Destination path

        Returns
        -------
        _: the list of all copied files.
        """
        from shutil import copytree

        source = self.DEFAULT_CONFIG_PATH
        new_directory: str = copytree(source, path, dirs_exist_ok=True)
        copied_files = [str(path) for path in Path(new_directory).rglob("*") if path.is_file()]

        return copied_files


# Load Melusine configuration
config = MelusineConfig()
config.reset()


class MelusineConfigError(Exception):
    """
    Exception raised when encountering config related errors.
    """

    CONST_CONFIG_ERROR_MESSAGE = f"""To modify the config use the `reset` method:
    - Using a dict:
      > from melusine import config
      > config.reset(config_dict=my_dict)
    - Using the path to a directory containing YAML files:
      > from melusine import config
      > config.reset(config_path=my_config_path)
    - Reset to default configurations:
      > from melusine import config
      > config.reset()
    - Using the {MelusineConfig.ENV_MELUSINE_CONFIG_DIR} environment variable:
      > import os
      > os.environ["{MelusineConfig.ENV_MELUSINE_CONFIG_DIR}"] = "/path/to/config/dir"
      > from melusine import config
      > config.reset()
    """

    def __init__(self, msg: str = CONST_CONFIG_ERROR_MESSAGE, *args: Any) -> None:
        """
        Initialize with a default error message.
        """
        super().__init__(msg, *args)
