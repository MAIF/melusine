import numpy as np
import pandas as pd
from typing import List

from transformers import TFAutoModel
from transformers import AutoTokenizer
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from melusine.models.models_v2.base_model import BaseMelusineModel


class TransformerMelusineModel(BaseMelusineModel):
    """
    Class to train (or fine-tune) transformer models.
    Just like all Melusine model classes, the `TransformerMelusineModel` class let's you include meta features.

    """

    # Attributes excluded from the default save methods
    # Ex: pickle.dump or json.dump
    EXCLUDE_LIST = ["tokenizer"]

    def __init__(
        self,
        text_column: str,
        model_name_or_path: str,
        seq_max: int = 128,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.text_column = text_column
        self.seq_max = seq_max
        self.tokenizer = None
        self.pkl_exclude.append("tokenizer")

    def fit(self, x: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Load the tokenizer and create the neural network.

        Parameters
        ----------
        x: pd.DataFrame
            Input features
        y: pd.Series
            Input labels
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.create_network()

    def transform(self, x: pd.DataFrame) -> List[np.ndarray]:
        """
        Encode the text input(s).
        Prepare the meta features.

        Parameters
        ----------
        x: pd.DataFrame
            Input features

        Returns
        -------
        inputs: List[np.ndarray]
            List of model inputs
        """
        inputs = list()

        # Encode text
        sequence_data = self.tokenizer.batch_encode_plus(
            x[self.text_column].to_list(),
            max_length=self.seq_max,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        inputs.append(np.asarray(sequence_data["input_ids"]))
        inputs.append(np.asarray(sequence_data["attention_mask"]))

        # Meta features
        inputs = self.get_meta_input(inputs, x)

        return inputs

    def create_network(self) -> None:
        """
        Create the neural network using Keras.
        This class can be easily customized using the following class attributes:
        - meta_archi
        - dense_archi
        - output_archi

        Returns
        -------

        """
        inputs = list()

        # Transformer inputs
        text_input = Input(shape=(self.seq_max,), dtype="int32")
        attention_input = Input(shape=(self.seq_max,), dtype="int32")
        inputs.append(text_input)
        inputs.append(attention_input)

        # Transformer Net
        base_transformer = TFAutoModel.from_pretrained(self.model_name_or_path)
        net = base_transformer(text_input, attention_input)[1]

        # Meta features net
        inputs, net = self.add_meta_net(inputs, net, meta_archi=self.meta_archi)

        # Final dense net
        dense_net = self.dense_archi(net)

        # Output layer
        output = self.output_archi(dense_net, self.n_targets)

        # Build model
        model = Model(inputs=inputs, outputs=output)

        self.model = model

    def save_model(self, path: str):
        super().save_model(path)

    def save_misc(self, path: str):
        """
        Save the tokenizer

        Parameters
        ----------
        path: str
            Save path
        """
        self.tokenizer.save_pretrained(path)
