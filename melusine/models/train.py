import ast
import numpy as np
import pickle

from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from melusine import config
from melusine.nlp_tools.text_processor import TextProcessor
from melusine.models.attention_model import PositionalEncoding
from melusine.models.attention_model import TransformerEncoderLayer
from melusine.models.attention_model import MultiHeadAttention

tensorboard_callback_parameters = config["tensorboard_callback"]


class NeuralModel(BaseEstimator, ClassifierMixin):
    """Generic class for  neural models.

    It is compatible with scikit-learn API (i.e. contains fit, transform
    methods).

    Parameters
    ----------
    neural_architecture_function : function,
        Function which returns a Model instance from Keras.
        Implemented model functions are: cnn_model, rnn_model, transformers_model, bert_model

    pretrained_embedding : np.array,
        Pretrained embedding matrix.

    text_input_column : str,
        Input text column to consider for the model.

    meta_input_list : list, optional
        List of the names of the columns containing the metadata.
        If empty list or None the model is used without metadata
        Default value, ['extension', 'dayofweek', 'hour', 'min'].

    vocab_size : int, optional
        Size of vocabulary for neurol network model.
        Default value, 25000.

    seq_size : int, optional
        Maximum size of input for neural model.
        Default value, 100.

    loss : str, optional
        Loss function for training.
        Default value, 'categorical_crossentropy'.

    activation : str, optional
        Activation function.
        Default value, 'softmax'.

    batch_size : int, optional
        Size of batches for the training of the neural network model.
        Default value, 4096.

    n_epochs : int, optional
        Number of epochs for the training of the neural network model.
        Default value, 15.

    bert_tokenizer : str, optional
        Tokenizer name from HuggingFace library or path to local tokenizer
        Only Camembert and Flaubert supported
        Default value, 'camembert-base'

    bert_model : str, optional
        Model name from HuggingFace library or path to local model

        Only Camembert and Flaubert supported
        Default value, 'camembert-base'

    Attributes
    ----------
    architecture_function, pretrained_embedding, text_input_column,
    meta_input_list, vocab_size, seq_size, loss, batch_size, n_epochs,

    model : Model instance from Keras,

    tokenizer : TextProcessor
        Tokenizer to split text into tokens

    embedding_matrix : np.array,
        Embedding matrix used as input for the neural network model.

    Examples
    --------
    >>> from melusine.models.train import NeuralModel
    >>> from melusine.models.neural_architectures import cnn_model
    >>> pretrained_embedding = MyEmbedding
    >>> list_meta = ['extension', 'dayofweek', 'hour']
    >>> nn_model = NeuralModel(cnn_model, pretrained_embedding, list_meta)
    >>> nn_model.fit(X_train, y_train)
    >>> y_res = nn_model.predict(X_test)

    """

    TOKENS_COL = "tokens"

    def __init__(
        self,
        pretrained_embedding=None,
        architecture_function=None,
        text_input_column="clean_text",
        meta_input_list=["extension", "dayofweek", "hour", "min"],
        vocab_size=25000,
        seq_size=100,
        embedding_dim=200,
        loss="categorical_crossentropy",
        activation="softmax",
        batch_size=4096,
        n_epochs=15,
        bert_tokenizer="jplu/tf-camembert-base",
        bert_model="jplu/tf-camembert-base",
        tokenizer=None,
        **kwargs,
    ):
        self.architecture_function = architecture_function
        # For retro-compatibility
        if hasattr(pretrained_embedding, "embedding"):
            pretrained_embedding = pretrained_embedding.embedding
        self.pretrained_embedding = pretrained_embedding
        if self.architecture_function.__name__ != "bert_model":
            if tokenizer is None:
                self.tokenizer = TextProcessor()
            else:
                self.tokenizer = tokenizer
        elif "camembert" in bert_tokenizer.lower():
            # Prevent the HuggingFace dependency
            try:
                from transformers import CamembertTokenizer

                self.tokenizer = CamembertTokenizer.from_pretrained(bert_tokenizer)
            except ModuleNotFoundError:
                raise (
                    """Please install transformers 3.4.0 (only version currently supported)
                    pip install melusine[transformers]"""
                )
        elif "flaubert" in bert_tokenizer.lower():
            # Prevent the HuggingFace dependency
            try:
                from transformers import XLMTokenizer

                self.tokenizer = XLMTokenizer.from_pretrained(bert_tokenizer)
            except ModuleNotFoundError:
                raise (
                    """Please install transformers 3.4.0 (only version currently supported)
                    pip install melusine[transformers]"""
                )
        else:
            raise NotImplementedError(
                "Bert tokenizer {} not implemented".format(bert_tokenizer)
            )
        self.text_input_column = text_input_column
        self.meta_input_list = meta_input_list
        self.vocab_size = vocab_size
        self.seq_size = seq_size
        self.embedding_dim = embedding_dim
        self.loss = loss
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.bert_model = bert_model
        self.nb_labels = 0
        self.nb_meta_features = 0
        self.vocabulary = []
        self.vocabulary_dict = {}

    def save_nn_model(self, filepath):
        """Save model to pickle, json and save weights to .h5."""

        if self.architecture_function.__name__ != "bert_model":
            json_model = self.model.to_json()
            open(filepath + ".json", "w").write(json_model)
            self.model.save_weights(filepath + "_model_weights.h5", overwrite=True)
        else:
            with open(filepath + "_bert_params.pkl", "wb") as f:
                pickle.dump([self.nb_labels, self.nb_meta_features], f)
            self.model.save_weights(filepath + "_model_weights", overwrite=True)
        pass

    def load_nn_model(self, filepath):
        """Save model from json and load weights from .h5."""

        if self.architecture_function.__name__ != "bert_model":
            model = model_from_json(
                open(filepath + ".json").read(),
                custom_objects={
                    "PositionalEncoding": PositionalEncoding,
                    "TransformerEncoderLayer": TransformerEncoderLayer,
                    "MultiHeadAttention": MultiHeadAttention,
                },
            )
            model.load_weights(filepath + "_model_weights.h5")
            model.compile(optimizer=Adam(), loss=self.loss, metrics=["accuracy"])
        else:
            with open(filepath + "_bert_params.pkl", "rb") as f:
                nb_labels, nb_meta_features = pickle.load(f)
            model = self.architecture_function(
                ntargets=nb_labels,
                seq_max=self.seq_size,
                nb_meta=nb_meta_features,
                loss=self.loss,
                activation=self.activation,
                bert_model=self.bert_model,
            )
            model.load_weights(filepath + "_model_weights")

        self.model = model
        pass

    def __getstate__(self):
        """Method called before serialization for a specific treatment to save
        model weight and structure instead of standard serialization."""
        dict_attr = dict(self.__dict__)
        if "model" in dict_attr:
            del dict_attr["model"]
            del dict_attr["embedding_matrix"]
            del dict_attr["pretrained_embedding"]
        return dict_attr

    def fit(
        self, X_train, y_train, tensorboard_log_dir=None, validation_data=None, **kwargs
    ):
        """Fit the neural network model on X and y.
        If meta_input list is empty list or None the model is used
        without metadata.

        Compatible with scikit-learn API.

        Parameters
        ----------
        X_train : pd.DataFrame

        y_train : pd.Series

        tensorboard_log_dir : str
            If not None, will be used as path to write logs for tensorboard
            Tensordboard callback parameters can be changed in config file

        validation_data: tuple
            Tuple of validation data
            Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be
            trained on this data. This could be a tuple (x_val, y_val). validation_data will override validation_split.
            Default value, None.

        Returns
        -------
        self : object
            Returns the instance
        """
        X_train = X_train.copy()
        y_train = y_train.copy()
        X_input_train, y_categorical_train = self._prepare_data(X_train, y_train)

        if validation_data:
            # Init X_val, y_val
            X_val, y_val = None, None

            try:
                X_val, y_val = validation_data
            except Exception as e:
                validation_data = None
                print(
                    "Validation_data has unexpected format. validation_data is now set to None. Following error:"
                    + str(e)
                )

            if validation_data:
                X_input_val, y_categorical_val = self._prepare_data(
                    X_val, y_val, validation_data=validation_data
                )
                validation_data = (X_input_val, y_categorical_val)

        if tensorboard_log_dir is None:
            self.model.fit(
                X_input_train,
                y_categorical_train,
                batch_size=self.batch_size,
                epochs=self.n_epochs,
                validation_data=validation_data,
                **kwargs,
            )
        else:

            histogram_freq = ast.literal_eval(
                tensorboard_callback_parameters["histogram_freq"]
            )
            write_graph = ast.literal_eval(
                tensorboard_callback_parameters["write_graph"]
            )
            write_grads = ast.literal_eval(
                tensorboard_callback_parameters["write_grads"]
            )
            write_images = ast.literal_eval(
                tensorboard_callback_parameters["write_images"]
            )
            embeddings_freq = ast.literal_eval(
                tensorboard_callback_parameters["embeddings_freq"]
            )
            embeddings_layer_names = ast.literal_eval(
                tensorboard_callback_parameters["embeddings_layer_names"]
            )
            embeddings_metadata = ast.literal_eval(
                tensorboard_callback_parameters["embeddings_metadata"]
            )
            embeddings_data = ast.literal_eval(
                tensorboard_callback_parameters["embeddings_data"]
            )

            if tensorboard_callback_parameters["update_freq"] in ["batch", "epoch"]:
                update_freq = tensorboard_callback_parameters["update_freq"]
            else:
                update_freq = ast.literal_eval(
                    tensorboard_callback_parameters["update_freq"]
                )

            tensorboard_callback = TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=histogram_freq,
                write_graph=write_graph,
                write_grads=write_grads,
                write_images=write_images,
                embeddings_freq=embeddings_freq,
                embeddings_layer_names=embeddings_layer_names,
                embeddings_metadata=embeddings_metadata,
                embeddings_data=embeddings_data,
                update_freq=update_freq,
            )
            self.model.fit(
                X_input_train,
                y_categorical_train,
                batch_size=self.batch_size,
                epochs=self.n_epochs,
                callbacks=[tensorboard_callback],
                validation_data=validation_data,
                **kwargs,
            )
        pass

    def predict(self, X, **kwargs):
        """Returns the class predicted.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        int
        """
        return np.argmax(self.predict_proba(X, **kwargs), axis=1)

    def predict_proba(self, X, **kwargs):
        """Returns the probabilities associated to each classes.
        If meta_input list is empty list or None the model is used
        without metadata.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.array
        """

        if self.architecture_function.__name__ != "bert_model":
            if isinstance(X, dict):
                X[self.TOKENS_COL] = self.tokenizer.process(X[self.text_input_column])
            else:
                X[self.TOKENS_COL] = X[self.text_input_column].apply(
                    self.tokenizer.process
                )
            X_seq = self._prepare_sequences(X)
            X_meta, nb_meta_features = self._get_meta(X)
            if nb_meta_features == 0:
                X_input = X_seq
            else:
                X_input = [X_seq, X_meta]
        else:
            X_seq, X_attention = self._prepare_bert_sequences(X)
            X_meta, nb_meta_features = self._get_meta(X)
            if nb_meta_features == 0:
                X_input = [X_seq, X_meta]
            else:
                X_input = [X_seq, X_attention, X_meta]
        return self.model.predict(X_input, **kwargs)

    def _create_vocabulary_from_tokens(self, X):
        """Create a word indexes dictionary from tokens."""
        token_series = X["tokens"]
        c = Counter([token for token_list in token_series for token in token_list])
        self.vocabulary = [t[0] for t in c.most_common(self.vocab_size)]
        pass

    def _get_embedding_matrix(self):
        """Prepares the embedding matrix to be used as an input for
        the neural network model.
        The vocabulary of the NN is those of the pretrained embedding
        """
        pretrained_embedding = self.pretrained_embedding
        self.vocabulary = pretrained_embedding.index_to_key
        vocab_size = len(self.vocabulary)
        vector_dim = pretrained_embedding.vector_size
        embedding_matrix = np.zeros((vocab_size + 2, vector_dim))
        for index, word in enumerate(self.vocabulary):
            if word not in ["PAD", "UNK"]:
                embedding_matrix[index + 2, :] = pretrained_embedding[word]
        embedding_matrix[1, :] = np.mean(embedding_matrix, axis=0)

        self.vocabulary.insert(0, "PAD")
        self.vocabulary.insert(1, "UNK")

        self.embedding_matrix = embedding_matrix
        pass

    def _generate_random_embedding_matrix(self):
        """Prepares the embedding matrix to be used as an input for
        the neural network model.
        The vocabulary of the NN is those of the pretrained embedding
        """
        vocab_size = len(self.vocabulary)
        vector_dim = self.embedding_dim
        embedding_matrix = np.random.uniform(
            low=-1, high=1, size=(vocab_size + 2, vector_dim)
        )
        embedding_matrix[0:2, :] = np.zeros((2, vector_dim))
        self.vocabulary.insert(0, "PAD")
        self.vocabulary.insert(1, "UNK")

        self.embedding_matrix = embedding_matrix
        pass

    def tokens_to_indices(self, tokens):
        """
        Input : list of tokens ["ma", "carte_verte", ...]
        Output : list of indices [46, 359, ...]
        """
        return [self.vocabulary_dict.get(token, 1) for token in tokens]

    def _prepare_sequences(self, X):
        """Prepares the sequence to be used as input for the neural network
        model.
        The input column must be an already tokenized text : tokens
        The tokens must have been obtained using the same tokenizer than the
        one used for the pre-trained embedding."""

        if isinstance(X, dict):
            seqs = [self.tokens_to_indices(X["tokens"])]
        else:
            seqs = X["tokens"].apply(self.tokens_to_indices)

        X_seq = pad_sequences(seqs, maxlen=self.seq_size)
        return X_seq

    def _prepare_bert_sequences(self, X):
        """Prepares the sequence to be used as input for the bert neural network
        model.
        The input column must  not be tokenized text : clean_text
        """

        if isinstance(X, dict):
            sequence = X[self.text_input_column]
        else:
            sequence = X[self.text_input_column].values.tolist()
        seqs = self.tokenizer.batch_encode_plus(
            sequence, max_length=self.seq_size, padding="max_length", truncation=True
        )

        return (
            np.asarray(seqs["input_ids"]),
            np.asarray(seqs["attention_mask"]),
        )

    def _prepare_data(self, X, y, validation_data=None):
        """Prepares the data for training and validation.
        1- Encodes y to categorical
        2- Differentiates data preparation for bert models and non-bert models
        3- Creates embedding matrix and init model for training data only (validation_data=None)

        Parameters
        ----------
        X : pd.DataFrame

        y : pd.Series

        validation_data : None or tuple
            Tuple of validation data
            Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be
            trained on this data. This could be a tuple (x_val, y_val). validation_data will override validation_split.
            Default value, None.

        Returns
        -------
        X_input : pd.DataFrame

        y_categorical : pd.Series
        """
        y_categorical = to_categorical(y)
        if not validation_data:
            nb_labels = len(np.unique(y))

        if self.architecture_function.__name__ != "bert_model":
            if isinstance(X, dict):
                X[self.TOKENS_COL] = self.tokenizer.process(X[self.text_input_column])
            else:
                X[self.TOKENS_COL] = X[self.text_input_column].apply(
                    self.tokenizer.process
                )
            X_meta, nb_meta_features = self._get_meta(X)

            if not validation_data:
                if self.pretrained_embedding:
                    self._get_embedding_matrix()
                else:
                    self._create_vocabulary_from_tokens(X)
                    self._generate_random_embedding_matrix()
                self.vocabulary_dict = {
                    word: i for i, word in enumerate(self.vocabulary)
                }
                self.model = self.architecture_function(
                    embedding_matrix_init=self.embedding_matrix,
                    ntargets=nb_labels,
                    seq_max=self.seq_size,
                    nb_meta=nb_meta_features,
                    loss=self.loss,
                    activation=self.activation,
                )

            X_seq = self._prepare_sequences(X)

            if nb_meta_features == 0:
                X_input = X_seq
            else:
                X_input = [X_seq, X_meta]

        else:
            X_seq, X_attention = self._prepare_bert_sequences(X)
            X_meta, nb_meta_features = self._get_meta(X)
            self.nb_labels, self.nb_meta_features = nb_labels, nb_meta_features
            if not validation_data:
                self.model = self.architecture_function(
                    ntargets=nb_labels,
                    seq_max=self.seq_size,
                    nb_meta=nb_meta_features,
                    loss=self.loss,
                    activation=self.activation,
                    bert_model=self.bert_model,
                )

            if nb_meta_features == 0:
                X_input = [X_seq, X_attention]
            else:
                X_input = [X_seq, X_attention, X_meta]

        return X_input, y_categorical

    def _get_meta(self, X):
        """Returns as a pd.DataFrame the metadata from X given the list_meta
        defined, and returns the number of columns. If meta_input_list is
        empty list or None, meta_input_list is returned as 0."""
        if self.meta_input_list is None or self.meta_input_list == []:
            X_meta = None
            nb_meta_features = 0
        else:
            meta_input_list = self.meta_input_list
            meta_input_list = [col + "__" for col in meta_input_list]

            if isinstance(X, dict):
                columns_list = list(X.keys())
            else:
                columns_list = list(X.columns)

            meta_columns_list = [
                col for col in columns_list if col.startswith(tuple(meta_input_list))
            ]

            if isinstance(X, dict):
                X_meta = np.array(
                    [
                        [X[meta_feature] for meta_feature in meta_columns_list],
                    ]
                )
            else:
                X_meta = X[meta_columns_list]

            nb_meta_features = len(meta_columns_list)

        return X_meta, nb_meta_features
