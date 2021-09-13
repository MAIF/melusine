import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Callable

from tensorflow import keras
from abc import ABC, abstractmethod
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Concatenate
from melusine.models.models_v2.default_nn_architectures import (
    default_meta_archi,
    default_dense_archi,
    default_multiclass_archi,
)

logger = logging.getLogger(__name__)


class BaseMelusineModel(ABC):
    """
    Base class for all Melusine models.
    Implements generic functions such as load, save, label encoding, etc
    """

    # Name of the pickle file created when saving
    PICKLE_FILENAME = "class.pkl"

    # Attributes excluded from the default save methods
    # Ex: pickle.dump or json.dump
    EXCLUDE_LIST = []

    def __init__(
        self,
        meta_input_list: List[str],
        meta_archi: Callable = default_meta_archi,
        dense_archi: Callable = default_dense_archi,
        output_archi: Callable = default_multiclass_archi,
    ):
        self.model = None
        self.label_encoder = None
        self.n_targets = None
        self.pkl_exclude = ["model"]
        self.meta_input_list = meta_input_list
        self.meta_cols = None
        self.meta_archi = meta_archi
        self.dense_archi = dense_archi
        self.output_archi = output_archi

    def __getstate__(self):
        """
        getstate is called when a class is pickled.
        Class attributes that should not be saved can be specified in the pkl_exclude list.
        For example, a pretrained embedding attribute should not be saved.

        Returns
        -------
        _: dict
            Instance dict to be saved
        """
        instance_dict = self.__dict__.copy()
        for key in self.EXCLUDE_LIST:
            instance_dict.pop(key)
        return instance_dict

    def fit_label_encoder(self, y: pd.Series) -> None:
        """
        Encode categorical labels.

        Parameters
        ----------
        y: pd.Series
            Labels
        """
        le = LabelEncoder()
        le.fit(y.astype(str))
        self.label_encoder = le
        self.n_targets = len(self.label_encoder.classes_)

        logger.info(f"Finished fitting label encoder ({self.n_targets} classes)")

    def fit_meta_cols(self, x: pd.DataFrame) -> None:

        if not self.meta_input_list:
            meta_cols = None
        else:
            meta_prefix_list = tuple(f"{col}__" for col in self.meta_input_list)
            meta_cols = [col for col in x.columns if col.startswith(meta_prefix_list)]

        self.meta_cols = meta_cols
        logger.info("Finished fitting meta columns")
        logger.info(f" Meta columns list : {self.meta_cols}")

    def get_meta_input(
        self, inputs: List[np.ndarray], x: pd.DataFrame
    ) -> List[np.ndarray]:

        if not self.meta_cols:
            return inputs

        inputs.append(x[self.meta_cols].values)
        return inputs

    def add_meta_net(
        self, inputs: List[np.ndarray], net, meta_archi: Callable
    ) -> (List[np.ndarray], str):  # TODO change

        # Meta features net
        if self.meta_cols:
            n_meta_features = len(self.meta_cols)
            meta_input, meta_net = meta_archi(n_meta_features)
            inputs.append(meta_input)
            net = Concatenate(axis=1)([net, meta_net])
            logger.info(f"Added meta net ({n_meta_features} meta features)")
        else:
            net = net
            logger.info(f"No meta features found, skipping meta net")

        return inputs, net

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series = None) -> None:
        """

        Parameters
        ----------
        x: pandas.DataFrame
            DataFrame with raw features
        y: pandas.Series
            Series with raw categories
        """
        self.fit_meta_cols(x)
        self.fit_label_encoder(y)

    @abstractmethod
    def transform(self, x: pd.DataFrame) -> List[np.ndarray]:
        pass

    def encode_labels(self, y: pd.Series) -> np.ndarray:
        return to_categorical(self.label_encoder.transform(y))

    def save_model(self, path: str) -> None:
        self.model.save(path)
        logger.info(f"Saved Deep Learning model to {path}")

    def save_misc(self, path: str) -> None:
        # Save anything that is not a model and shouldn't be pickled
        pass

    def load_misc(self, path: str) -> None:
        # Load anything that is not a model and shouldn't be pickled
        pass

    def save(self, path: str) -> None:
        self.save_model(path)
        self.save_misc(path)
        pkl_path = os.path.join(path, self.PICKLE_FILENAME)
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved Melusine model to {path}")

    @classmethod
    def load(cls, path: str) -> None:
        pkl_path = os.path.join(path, cls.PICKLE_FILENAME)
        with open(pkl_path, "rb") as f:
            instance = pickle.load(f)

        instance.model = keras.models.load_model(path)
        logger.info(f"Loaded Melusine model from {path}")

        instance.load_misc(path)

        return instance
