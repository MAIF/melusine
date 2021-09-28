import os
import json
import glob
import logging
import importlib
import pickle
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseMelusineClass(ABC):
    EXCLUDE_LIST = list()
    CONFIG_KEY = None

    # JSON save config
    SORT_KEYS = True
    INDENT = 4

    def __init__(self):
        pass

    @property
    @abstractmethod
    def FILENAME(self):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str, filename_prefix: str = None) -> None:
        raise NotImplementedError()

    def get_file_path(self, filename_prefix, path):
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
            (filename_prefix + "_" if filename_prefix else "") + self.FILENAME,
        )
        return full_path

    @classmethod
    def search_file(cls, path):
        # Look for candidate files at given path
        suffix = cls.FILENAME
        candidate_files = [x for x in glob.glob(os.path.join(path, f"*{suffix}"))]
        if not candidate_files:
            raise FileNotFoundError(f"Could not find {suffix} file at path {path}")
        elif len(candidate_files) == 1:
            filepath = candidate_files[0]
            logger.info(f"Reading file {filepath}")
        else:
            raise FileNotFoundError(
                f"Found multiple files ending with {suffix} : {candidate_files}"
            )

        return filepath

    @classmethod
    def load_json(cls, path):
        filepath = cls.search_file(path)
        with open(filepath, "r") as f:
            data = json.load(f)

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

        full_path = self.get_file_path(filename_prefix, path)
        with open(full_path, "w") as f:
            json.dump(save_dict, f, sort_keys=self.SORT_KEYS, indent=self.INDENT)

    @classmethod
    def load_pkl(cls, path):
        filepath = cls.search_file(path)
        with open(filepath, "rb") as f:
            instance = pickle.load(f)

        return instance

    def save_pkl(self, path, filename_prefix):
        full_path = self.get_file_path(filename_prefix, path)

        with open(full_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def save_obj(config_dict, path, filename_prefix, config_key):
        obj = config_dict.pop(config_key, None)
        if obj:
            config_dict[f"{config_key}"] = True
            config_dict[f"{config_key}_name"] = type(obj).__name__
            config_dict[f"{config_key}_module"] = type(obj).__module__
            obj.save(path, filename_prefix)

    @staticmethod
    def load_obj(config_dict, path, obj_key):
        if config_dict.pop(obj_key):
            obj_class_name = config_dict.pop(f"{obj_key}_name")
            obj_module = config_dict.pop(f"{obj_key}_module")
            obj_class = BaseMelusineClass.load_class_obj(
                obj_name=obj_class_name, obj_path=obj_module
            )
            obj = obj_class.load(path)
        else:
            obj = None

        return obj

    @staticmethod
    def load_class_obj(obj_path, obj_name):
        module_obj = importlib.import_module(obj_path)
        if not hasattr(module_obj, obj_name):
            raise AttributeError(
                f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
            )
        return getattr(module_obj, obj_name)
