import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam


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

    tokenizer : Tokenizer instance from Keras,

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
                 architecture_function=None,
                 pretrained_embedding=None,
                 text_input_column='clean_text',
                 meta_input_list=['extension', 'dayofweek', 'hour', 'min'],
                 vocab_size=25000,
                 seq_size=100,
                 loss='categorical_crossentropy',
                 batch_size=4096,
                 n_epochs=15,
                 **kwargs):
        self.architecture_function = architecture_function
        self.pretrained_embedding = pretrained_embedding
        self.text_input_column = text_input_column
        self.meta_input_list = meta_input_list
        self.vocab_size = vocab_size
        self.seq_size = seq_size
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

    def fit(self, X, y, **kwargs):
        """Fit the neural network model on X and y.
        If meta_input list is empty list or None the model is used
        without metadata.

        Compatible with scikit-learn API.

        Parameters
        ----------
        X : pd.DataFrame

        y : pd.Series

        Returns
        -------
        self : object
            Returns the instance
        """
        self._fit_tokenizer(X)
        self._create_word_indexes_from_tokens()
        self._get_embedding_matrix()

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

        self.model.fit(X_input,
                       y_categorical,
                       batch_size=self.batch_size,
                       epochs=self.n_epochs,
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
        X_seq = self._prepare_sequences(X)
        X_meta, nb_meta_features = self._get_meta(X)
        if nb_meta_features == 0:
            X_input = X_seq
        else:
            X_input = [X_seq, X_meta]
        return self.model.predict(X_input, **kwargs)

    def _fit_tokenizer(self, X):
        """Fit a Tokenizer instance from Keras on a clean body."""
        self.tokenizer = Tokenizer(num_words=self.vocab_size,
                                   oov_token='UNK')
        self.tokenizer.fit_on_texts(X[self.text_input_column])
        pass

    def _create_word_indexes_from_tokens(self):
        """Create a word indexes dictionary from tokens."""
        c = Counter(self.tokenizer.word_counts)
        self.tokenizer.word_index = {t[0]: i + 1 for i, t
                                     in enumerate(c.most_common(len(c)))}
        self.tokenizer.word_index['UNK'] = 0
        pass

    def _get_embedding_matrix(self):
        """Prepares the embedding matrix to be used as an input for
        the neural network model."""
        pretrained_embedding = self.pretrained_embedding
        vocab_size = self.vocab_size
        vector_dim = pretrained_embedding.embedding.vector_size
        wv_dict = {word: vec for word, vec in
                   zip(pretrained_embedding.embedding.wv.index2word,
                       pretrained_embedding.embedding.wv.syn0)}
        embedding_matrix = np.zeros((vocab_size+1, vector_dim))
        for word, index in self.tokenizer.word_index.items():
            if index >= vocab_size:
                continue
            embedding_vector = wv_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        self.embedding_matrix = embedding_matrix
        pass

    def _prepare_sequences(self, X):
        """Prepares the sequence to be used as input for the neural network
        model."""
        seqs = self.tokenizer.texts_to_sequences(X[self.text_input_column])
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
            columns_list = list(X.columns)
            meta_columns_list = [col for col in columns_list
                                 if col.startswith(tuple(meta_input_list))]
            X_meta = X[meta_columns_list]
            nb_meta_features = len(meta_columns_list)

        return X_meta, nb_meta_features
