import os
import json
import glob
import logging
import importlib
import pickle
from abc import ABC, abstractmethod

from melusine import config

logger = logging.getLogger(__name__)


class SaverMixin(ABC):
    EXCLUDE_LIST = ["func"]

    # Save params
    SAVE_NAME = "name"
    SAVE_CLASS = "class_name"
    SAVE_MODULE = "module"

    # JSON save params
    JSON_SUFFIX = ".json"
    SORT_KEYS = True
    INDENT = 4

    # Pickle save params
    PKL_SUFFIX = ".pkl"

    # Config Mapping
    CONFIG_MARKER = "_CONFIG"
    CONFIG_KEY = "config_key"
    CONFIG_MAPPING = "config_mapping"

    def __init__(self, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, filename_prefix: str = None):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str, filename_prefix: str = None) -> None:
        raise NotImplementedError()

    @staticmethod
    def get_file_path(filename, path, filename_prefix):

        # Check if path is a file
        if os.path.isfile(path):
            raise AttributeError(
                f"Provided path ({path}) should be a directory, not a file"
            )

        # Create directory if necessary
        os.makedirs(path, exist_ok=True)

        # Save the tokenizer config file
        full_path = os.path.join(
            path,
            (filename_prefix + "_" if filename_prefix else "") + filename,
        )
        return full_path

    @staticmethod
    def search_file(filename, path, filename_prefix: str = None):
        # Look for candidate files at given path
        if filename_prefix:
            pattern = f"{filename_prefix}_{filename}"
        else:
            pattern = f"{filename}"

        candidate_files = [x for x in glob.glob(os.path.join(path, pattern))]
        if not candidate_files:
            raise FileNotFoundError(
                f"Could not find files matching pattern {pattern} at path {path}"
            )
        elif len(candidate_files) == 1:
            filepath = candidate_files[0]
            logger.info(f"Reading file {filepath}")
        else:
            raise FileNotFoundError(
                f"Found multiple files matching with  pattern {pattern} : {candidate_files}"
            )

        return filepath

    @classmethod
    def load_json(cls, path, filename_prefix: str = None):
        filename = cls.__name__ + cls.JSON_SUFFIX
        filepath = cls.search_file(filename, path, filename_prefix=filename_prefix)
        with open(filepath, "r") as f:
            data = json.load(f)

        # data = cls.parse_config_parameters(data)

        return data

    def save_json(
        self,
        save_dict,
        path,
        filename_prefix,
    ):
        # Pop items from the exclude list
        for key in self.EXCLUDE_LIST:
            _ = save_dict.pop(key, None)

        self.save_config_mapping(save_dict)

        # Convert sets to lists (json compatibility)
        for key, val in save_dict.items():
            if isinstance(val, set):
                save_dict[key] = list(val)

        # Save file
        filename = type(self).__name__ + self.JSON_SUFFIX
        full_path = self.get_file_path(filename, path, filename_prefix)
        with open(full_path, "w") as f:
            json.dump(save_dict, f, sort_keys=self.SORT_KEYS, indent=self.INDENT)

    def save_config_mapping(self, save_dict):
        config_mapping = save_dict.pop(self.CONFIG_MAPPING, dict())
        if not config_mapping:
            return save_dict

        config_key = getattr(self, self.CONFIG_KEY, None)
        if not config_key:
            raise AttributeError(
                f"Error: Found {self.CONFIG_MAPPING} attribute without a {self.CONFIG_KEY} attribute."
            )

        for key, value in config_mapping.items():
            save_dict.pop(key)
            save_dict[key] = f"{self.CONFIG_MARKER}.{config_key}.{key}"

    @classmethod
    def load_pkl(cls, path, filename_prefix: str = None):
        filename = cls.__name__ + cls.PKL_SUFFIX
        return cls.load_pkl_generic(filename, path, filename_prefix=filename_prefix)

    @staticmethod
    def load_pkl_generic(filename, path, filename_prefix: str = None):
        # Add pickle suffix if necessary
        if not filename.endswith(SaverMixin.PKL_SUFFIX):
            filename += SaverMixin.PKL_SUFFIX

        filepath = SaverMixin.search_file(
            filename, path, filename_prefix=filename_prefix
        )
        with open(filepath, "rb") as f:
            instance = pickle.load(f)

        return instance

    def save_pkl(self, path, filename_prefix):
        filename = type(self).__name__ + self.PKL_SUFFIX
        full_save_path = self.save_pkl_generic(self, filename, path, filename_prefix)

        return full_save_path

    @staticmethod
    def save_pkl_generic(obj, filename, path, filename_prefix):
        # Add pickle suffix if necessary
        if not filename.endswith(SaverMixin.PKL_SUFFIX):
            filename += SaverMixin.PKL_SUFFIX

        full_path = SaverMixin.get_file_path(filename, path, filename_prefix)

        with open(full_path, "wb") as f:
            pickle.dump(obj, f)

        return full_path

    @staticmethod
    def get_obj_meta(obj, name):
        return {
            SaverMixin.SAVE_NAME: name,
            SaverMixin.SAVE_CLASS: type(obj).__name__,
            SaverMixin.SAVE_MODULE: type(obj).__module__,
        }

    @staticmethod
    def load_obj(obj_params, path, filename_prefix: str = None):
        obj_name = obj_params[SaverMixin.SAVE_NAME]
        obj_class_name = obj_params[SaverMixin.SAVE_CLASS]
        obj_module = obj_params[SaverMixin.SAVE_MODULE]
        obj_class = SaverMixin.load_class_obj(
            obj_name=obj_class_name, obj_path=obj_module
        )

        # Object has a load method
        if hasattr(obj_class, "load"):
            obj = obj_class.load(path, filename_prefix=filename_prefix)
        # Use pickle load
        else:
            obj = SaverMixin.load_pkl_generic(
                obj_name, path, filename_prefix=filename_prefix
            )

        return obj_name, obj

    @staticmethod
    def load_class_obj(obj_path, obj_name):
        module_obj = importlib.import_module(obj_path)
        if not hasattr(module_obj, obj_name):
            raise AttributeError(
                f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
            )
        return getattr(module_obj, obj_name)

    @classmethod
    def from_config(cls, config_key, **kwargs):
        init_params = dict()

        # Add parameters from the conf
        params_from_config = config[config_key]
        init_params.update(**params_from_config)

        # Add keyword arguments
        init_params.update(**kwargs)

        # Instantiate object
        instance = cls(**init_params)

        # Config Mapping parameters
        setattr(instance, cls.CONFIG_MAPPING, params_from_config)
        setattr(instance, cls.CONFIG_KEY, config_key)

        return instance

    @classmethod
    def from_json(cls, **params_dict):
        config_key = params_dict.pop(cls.CONFIG_KEY, None)
        if config_key:
            init_params = dict()
            for key, value in params_dict.items():
                if not cls.is_from_config(value):
                    init_params[key] = value
            return cls.from_config(config_key, **init_params)
        else:
            return cls(**params_dict)

    @classmethod
    def is_from_config(cls, value):
        if isinstance(value, str) and value.startswith(cls.CONFIG_MARKER):
            return True
