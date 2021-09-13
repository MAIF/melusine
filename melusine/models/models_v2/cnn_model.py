import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from melusine.models.models_v2.base_model import BaseMelusineModel
from melusine.nlp_tools.tokenizer import WordLevelTokenizer
from melusine.models.models_v2.default_nn_architectures import (
    default_cnn_archi,
    default_meta_archi,
    default_dense_archi,
    default_multiclass_archi,
)

logger = logging.getLogger(__name__)


class CnnMelusineModel(BaseMelusineModel):
    """
    The CnnMelusineModel class let's you create, load and save a Keras CNN model.
    The CNN definition is modular and users may easily adapt the following parameters:
    - CNN network architecture
    - Meta data network architecture
    - Dense network architecture (aggregation of CNN and Meta)
    - Output layer (multi-class, multi-lable, etc)
    The CnnMelusineModel inherits from the generic class BaseMelusineModel.
    """

    # Name of the tokenizer file when saving the model
    TOKENIZER_FILENAME = "tokenizer.json"

    # Attributes excluded from the default save methods
    # Ex: pickle.dump or json.dump
    EXCLUDE_LIST = ["pretrained_embedding", "tokenizer"]

    def __init__(
        self,
        text_column,
        tokenizer,
        seq_max,
        pretrained_embedding,
        meta_input_list=None,
        unk_token="[UNK]",
        pad_token="[PAD]",
        trainable=False,
        cnn_archi: Callable = default_cnn_archi,
        meta_archi: Callable = default_meta_archi,
        dense_archi: Callable = default_dense_archi,
        output_archi: Callable = default_multiclass_archi,
    ):
        """

        Parameters
        ----------
        text_column: str
            DataFrame colum containing the text data
        tokenizer: WordLevelTokenizer
            Tokenizer object
        seq_max: int
            Maximum length of the sequence of tokens fed to the model
        pretrained_embedding: gensim.Embedding
            Pretrained word-embeddings
        meta_input_list: List[str]
            List of metadata features
        unk_token: str
            Value of the unknown token
        pad_token: str
            Value of the padding token
        trainable: bool
            If True: train the embeddings weights together with the model
        cnn_archi: Callable
            Function that defines a CNN architecture
        """
        super().__init__(
            meta_input_list=meta_input_list,
            meta_archi=meta_archi,
            dense_archi=dense_archi,
            output_archi=output_archi,
        )

        self.seq_max = seq_max
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.trainable = trainable
        if hasattr(pretrained_embedding, "wv"):
            pretrained_embedding = pretrained_embedding.wv
        self.pretrained_embedding = pretrained_embedding
        self.vocab = None
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.unk_id = None
        self.cnn_archi = cnn_archi

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert
        Parameters
        ----------
        tokens: List[str]
            List of tokens

        Returns
        -------
        ids: List[int]
            List of IDs
        """
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens]
        return ids

    def fit(self, x: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Build the vocabulary and create the neural network.

        Parameters
        ----------
        x: pd.DataFrame
            Input features
        y: pd.Series
            Input labels
        """
        super().fit(x, y)
        self.vocab = self._get_embedding_vocab()
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
        x_input = x.copy()
        inputs = list()

        # Fill NAs in text column
        text_data = x_input[self.text_column].fillna("").astype(str)

        # Convert text to List[token]
        token_data = text_data.apply(self.tokenizer.tokenize)

        # Convert List[token] to List[ids]
        id_data = token_data.apply(self.tokens_to_ids)

        # Padding : Convert List[ids] to padded List[ids]
        sequence_data = pad_sequences(id_data, maxlen=self.seq_max, value=self.pad_id)
        inputs.append(sequence_data)

        # Meta features
        inputs = self.get_meta_input(inputs, x)

        return inputs

    def _get_embedding_vocab(self) -> Dict[str, int]:
        """
        Get the embedding vocabulary to match token IDs with vectors.

        Returns
        -------
        vocab: Dict[str, int]
        """
        pretrained_embedding = self.pretrained_embedding
        embedding_dim = pretrained_embedding.vector_size

        if self.pad_token not in pretrained_embedding.index2word:
            pretrained_embedding[self.pad_token] = np.zeros(embedding_dim)
        if self.unk_token not in pretrained_embedding.index2word:
            pretrained_embedding[self.unk_token] = np.zeros(embedding_dim)

        self.unk_id = pretrained_embedding.index2word.index(self.unk_token)
        self.pad_id = pretrained_embedding.index2word.index(self.pad_token)

        vocab = {tok: i for i, tok in enumerate(pretrained_embedding.wv.index2word)}
        logger.info(f"Finished building vocabulary ({len(vocab)} tokens in vocab)")

        return vocab

    def create_network(self) -> None:
        """
        Create the neural network using Keras.
        This class can be easily customized using the following class attributes:
        - cnn_archi
        - meta_archi
        - dense_archi
        - output_archi

        Returns
        -------
        model: Model
            Keras Model to be trained on data
        """
        logger.info("Start creating CNN network")
        inputs = list()

        # Text input
        text_input = Input(shape=(self.seq_max,), dtype="int32")
        inputs.append(text_input)

        # Embedding layer
        embedding_net = self.pretrained_embedding.get_keras_embedding(
            train_embeddings=self.trainable
        )(text_input)

        # CNN net
        net = self.cnn_archi(embedding_net)

        # Meta features net
        inputs, net = self.add_meta_net(inputs, net, meta_archi=self.meta_archi)

        # Final dense net
        dense_net = self.dense_archi(net)

        # Output layer
        output = self.output_archi(dense_net, self.n_targets)

        # Build model
        model = Model(inputs=inputs, outputs=output)

        self.model = model
        logger.info("Finished creating CNN network")

    def save_model(self, path: str):
        """
        Regular Keras model saving, the base class does it well.

        Parameters
        ----------
        path: str
            Save path
        """
        super().save_model(path)

    def save_misc(self, path: str):
        """
        Save the tokenizer

        Parameters
        ----------
        path: str
            Save path
        """
        tokenizer_path = os.path.join(path, self.TOKENIZER_FILENAME)
        self.tokenizer.save(tokenizer_path)

    def load_misc(self, path: str) -> None:
        """
        Load the tokenizer

        Parameters
        ----------
        path: str
            Load path
        """
        # Load tokenizer
        tokenizer_path = os.path.join(path, self.TOKENIZER_FILENAME)
        self.tokenizer = WordLevelTokenizer.load(tokenizer_path)
