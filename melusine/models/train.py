from melusine.nlp_tools.tokenizer import Tokenizer
import numpy as np
import ast
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from melusine.utils.transformer_scheduler import TransformerScheduler
from melusine.config.config import ConfigJsonReader

conf_reader = ConfigJsonReader()
config = conf_reader.get_config_file()
tensorboard_callback_parameters = config["tensorboard_callback"]

class NeuralModel(BaseEstimator, ClassifierMixin):
    """Generic class for  neural models.

    It is compatible with scikit-learn API (i.e. contains fit, transform
    methods).

    Parameters
    ----------
    neural_architecture_function : function,
        Function which returns a Model instance from Keras.

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

    batch_size : int, optional
        Size of batches for the training of the neural network model.
        Default value, 4096.

    n_epochs : int, optional
        Number of epochs for the training of the neural network model.
        Default value, 15.

    Attributes
    ----------
    architecture_function, pretrained_embedding, text_input_column,
    meta_input_list, vocab_size, seq_size, loss, batch_size, n_epochs,

    model : Model instance from Keras,

    tokenizer : Tokenizer instance from Melusine,

    embedding_matrix : np.array,
        Embedding matrix used as input for the neural network model.

    Examples
    --------
    >>> from melusine.models.train import NeuralModel
    >>> from melusine.models.neural_architectures import cnn_model
    >>> from melusine.nlp_tools.embedding import Embedding
    >>> pretrained_embedding = Embedding.load()
    >>> list_meta = ['extension', 'dayofweek', 'hour']
    >>> nn_model = NeuralModel(cnn_model, pretrained_embedding, list_meta)
    >>> nn_model.fit(X_train, y_train)
    >>> y_res = nn_model.predict(X_test)

    """

    def __init__(self,
                pretrained_embedding=None,
                 architecture_function=None,
                 text_input_column='clean_text',
                 meta_input_list=['extension', 'dayofweek', 'hour', 'min'],
                 vocab_size=25000,
                 seq_size=100,
                 embedding_dim=200,
                 loss='categorical_crossentropy',
                 batch_size=4096,
                 n_epochs=15,
                 **kwargs):
        self.architecture_function = architecture_function
        self.pretrained_embedding = pretrained_embedding
        self.tokenizer = Tokenizer(input_column = text_input_column)
        self.text_input_column = text_input_column
        self.meta_input_list = meta_input_list
        self.vocab_size = vocab_size
        self.seq_size = seq_size
        self.embedding_dim = embedding_dim
        self.loss = loss
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def save_nn_model(self, filepath):
        """Save model to pickle, json and save weights to .h5."""
        json_model = self.model.to_json()
        open(filepath+".json", 'w').write(json_model)
        self.model.save_weights(filepath+"_model_weights.h5", overwrite=True)
        pass

    def load_nn_model(self, filepath):
        """Save model from json and load weights from .h5."""
        model = model_from_json(open(filepath+".json").read())
        model.load_weights(filepath+"_model_weights.h5")
        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        pass

    def __getstate__(self):
        """Method called before serialization for a specific treatment to save
        model weight and structure instead of standard serialization."""
        dict_attr = dict(self.__dict__)
        if 'model' in dict_attr:
            del dict_attr["model"]
            del dict_attr["embedding_matrix"]
            del dict_attr["pretrained_embedding"]
        return dict_attr

    def __setstate__(self, dict_attr):
        """Method called before loading class for a specific treatment to load
        model weight and structure instead of standard serialization."""
        self.__dict__ = dict_attr

    def fit(self, X, y,tensorboard_log_dir=None, **kwargs):
        """Fit the neural network model on X and y.
        If meta_input list is empty list or None the model is used
        without metadata.

        Compatible with scikit-learn API.

        Parameters
        ----------
        X : pd.DataFrame

        y : pd.Series

        tensorboard_log_dir : str
            If not None, will be used as path to write logs for tensorboard
            Tensordboard callback parameters can be changed in config file

        Returns
        -------
        self : object
            Returns the instance
        """

        X = self.tokenizer.transform(X)
        if self.pretrained_embedding:
            self._get_embedding_matrix()
        else:
            self._create_vocabulary_from_tokens(X)
            self._generate_random_embedding_matrix()
        self.vocabulary_dict = {word: i for i, word in enumerate(self.vocabulary)}
        X_seq = self._prepare_sequences(X)
        X_meta, nb_meta_features = self._get_meta(X)
        y_categorical = np_utils.to_categorical(y)
        nb_labels = len(np.unique(y))

        self.model = self.architecture_function(
            embedding_matrix_init=self.embedding_matrix,
            ntargets=nb_labels,
            seq_max=self.seq_size,
            nb_meta=nb_meta_features,
            loss=self.loss)

        if nb_meta_features == 0:
            X_input = X_seq
        else:
            X_input = [X_seq, X_meta]

        if tensorboard_log_dir is None:
            self.model.fit(X_input,
                           y_categorical,
                           batch_size=self.batch_size,
                           epochs=self.n_epochs,
                           **kwargs)
        else:

            histogram_freq = ast.literal_eval(tensorboard_callback_parameters["histogram_freq"])
            batch_size = ast.literal_eval(tensorboard_callback_parameters["batch_size"])
            write_graph = ast.literal_eval(tensorboard_callback_parameters["write_graph"])
            write_grads = ast.literal_eval(tensorboard_callback_parameters["write_grads"])
            write_images = ast.literal_eval(tensorboard_callback_parameters["write_images"])
            embeddings_freq = ast.literal_eval(tensorboard_callback_parameters["embeddings_freq"])
            embeddings_layer_names = ast.literal_eval(tensorboard_callback_parameters["embeddings_layer_names"])
            embeddings_metadata = ast.literal_eval(tensorboard_callback_parameters["embeddings_metadata"])
            embeddings_data = ast.literal_eval(tensorboard_callback_parameters["embeddings_data"])

            if tensorboard_callback_parameters["update_freq"] in ['batch', 'epoch']:
                update_freq = tensorboard_callback_parameters["update_freq"]
            else:
                update_freq = ast.literal_eval(tensorboard_callback_parameters["update_freq"])



            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir,
                                               histogram_freq=histogram_freq,
                                               batch_size=batch_size,
                                               write_graph=write_graph,
                                               write_grads=write_grads,
                                               write_images=write_images,
                                               embeddings_freq=embeddings_freq,
                                               embeddings_layer_names=embeddings_layer_names,
                                               embeddings_metadata=embeddings_metadata,
                                               embeddings_data=embeddings_data,
                                               update_freq=update_freq)
            self.model.fit(X_input,
                           y_categorical,
                           batch_size=self.batch_size,
                           epochs=self.n_epochs,
                           callbacks=[tensorboard_callback],
                           **kwargs)
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
        X = self.tokenizer.transform(X)
        X_seq = self._prepare_sequences(X)
        X_meta, nb_meta_features = self._get_meta(X)
        if nb_meta_features == 0:
            X_input = X_seq
        else:
            X_input = [X_seq, X_meta]
        return self.model.predict(X_input, **kwargs)

    def _create_vocabulary_from_tokens(self, X):
        """Create a word indexes dictionary from tokens."""
        token_series = X['tokens']
        c = Counter([token for token_list in token_series for token in token_list])
        self.vocabulary = [t[0] for t in c.most_common(self.vocab_size)]
        pass

    def _get_embedding_matrix(self):
        """Prepares the embedding matrix to be used as an input for
        the neural network model.
        The vocabulary of the NN is those of the pretrained embedding
        """
        pretrained_embedding = self.pretrained_embedding
        self.vocabulary = pretrained_embedding.embedding.wv.index2word
        vocab_size = len(self.vocabulary)
        vector_dim = pretrained_embedding.embedding.vector_size
        embedding_matrix = np.zeros((vocab_size + 2, vector_dim))
        for index, word in enumerate(self.vocabulary):
            if word not in ['PAD', 'UNK']:
                embedding_matrix[index + 2, :] = pretrained_embedding.embedding.wv.get_vector(word)
        embedding_matrix[1, :] = np.mean(embedding_matrix, axis=0)

        self.vocabulary.insert(0, 'PAD')
        self.vocabulary.insert(1, 'UNK')

        self.embedding_matrix = embedding_matrix
        pass

    def _generate_random_embedding_matrix(self):
        """Prepares the embedding matrix to be used as an input for
        the neural network model.
        The vocabulary of the NN is those of the pretrained embedding
        """
        vocab_size = len(self.vocabulary)
        vector_dim = self.embedding_dim
        embedding_matrix = np.random.uniform(low=-1,high=1,size=(vocab_size + 2, vector_dim))
        embedding_matrix[0:2, :] = np.zeros((2, vector_dim))
        self.vocabulary.insert(0, 'PAD')
        self.vocabulary.insert(1, 'UNK')

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
        The tokens must have been optained using the same tokenizer than the
        one used for the pre-trained embedding."""

        if isinstance(X, dict):
            seqs = [self.tokens_to_indices(X['tokens'])]
        else:
            seqs = X['tokens'].apply(self.tokens_to_indices)

        X_seq = pad_sequences(seqs, maxlen=self.seq_size)
        return X_seq

    def _get_meta(self, X):
        """Returns as a pd.DataFrame the metadata from X given the list_meta
        defined, and returns the number of columns. If meta_input_list is
        empty list or None, meta_input_list is returned as 0."""
        if self.meta_input_list is None or self.meta_input_list == []:
            X_meta = None
            nb_meta_features = 0
        else:
            meta_input_list = self.meta_input_list
            meta_input_list = [col+'__' for col in meta_input_list]

            if isinstance(X, dict):
                columns_list = list(X.keys())
            else:
                columns_list = list(X.columns)

            meta_columns_list = [col for col in columns_list if col.startswith(tuple(meta_input_list))]

            if isinstance(X, dict):
                X_meta = np.array([[X[meta_feature] for meta_feature in meta_columns_list], ])
            else:
                X_meta = X[meta_columns_list]

            nb_meta_features = len(meta_columns_list)

        return X_meta, nb_meta_features
