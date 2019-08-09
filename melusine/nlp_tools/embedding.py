import logging
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Vocab, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
# from glove import Corpus, Glove

from melusine.utils.streamer import Streamer

log = logging.getLogger('Embeddings')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%d/%m %I:%M')


class Embedding:
    """Class to train embeddings with Word2Vec algorithm.
    Attributes
    ----------
    input_column : str,
        Input text column to consider for the embedding.
    stream : Streamer instance,
        Builds a stream a tokens from a pd.Dataframe to train the embeddings.
    embedding : Word2Vec instance from Gensim
    Examples
    --------
    >>> from melusine.nlp_tools.embedding import Embedding
    >>> embedding = Embedding()
    >>> embedding.train(X)
    >>> embedding.save(filepath)
    >>> embedding = Embedding().load(filepath)
    """

    def __init__(self,
                 input_column='clean_text',
                 workers=40,
                 seed=42,
                 iter=15,
                 size=300,
                 window=5,
                 min_count=100,
                 stop_removal = True):

        self.logger = logging.getLogger('NLUtils.Embedding')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.debug('Create an Embedding instance.')
        self.input_column = input_column
        self.stop_removal = stop_removal
        self.streamer = Streamer(column=self.input_column, stop_removal=stop_removal)
        self.workers = workers
        self.seed = seed
        self.iter = iter
        self.size = size
        self.window = window
        self.min_count = min_count
        self.word2id = {}
        self.embedding = None
        self.kv = None

    def save(self, filepath):
        """Method to save Embedding object."""
        self.embedding.save(filepath)

    def load(self, filepath):
        """Method to load Embedding object."""
        self.embedding = KeyedVectors.load(filepath, mmap='r')
        return self

    def train(self, X, embedding_type='word2vec'):
        """Train embeddings with the desired word embedding algorithm (default is Word2Vec).
        Parameters
        ----------
        X : pd.Dataframe
            Containing a clean body column.
        embedding_type: str
            Desired embedding type
        Returns
        -------
        self : object
            Returns the instance
        """
        self.logger.info('Start training for embedding')

        self.streamer.to_stream(X)

        if embedding_type == 'word2vec':
            self.train_word2vec()
        elif embedding_type == 'tfidf':
            self.train_tfidf(X, self.input_column)
        elif embedding_type == 'docterm':
            self.train_docterm(X, self.input_column)
        #elif embedding_type == 'glove':
        #    self.train_glove(X, self.input_column)

        self.logger.info('Done.')

    def train_tfidf(self, X, input_column):
        """TODO
        Parameters
        ----------
        X : pd.Dataframe
            Containing a column with tokens.
        input_column: str
            Name of the input column containing tokens
        """

        def dummy_function(doc):
            return doc

        tfidf_vec = TfidfVectorizer(
            analyzer='word',
            tokenizer=dummy_function,
            preprocessor=dummy_function,
            token_pattern=None,
            use_idf=True,
            norm='l2'
        )

        tfidf_data = tfidf_vec.fit_transform(X[input_column])

        self.word2id = tfidf_vec.vocabulary_
        embedding_matrix = self.train_svd(tfidf_data)

        self.create_keyedvector_from_matrix(embedding_matrix, self.word2id)

    def train_docterm(self, X, input_column):
        """TODO
        Parameters
        ----------
        X : pd.Dataframe
            Containing a column with tokens.
        input_column: str
            Name of the input column containing tokens
        """

        def dummy_function(doc):
            return doc

        # CountVectorizer

        count_vec = CountVectorizer(
            analyzer='word',
            tokenizer=dummy_function,
            preprocessor=dummy_function,
            token_pattern=None
        )

        docterm_data = count_vec.fit_transform(X[input_column])

        self.word2id = count_vec.vocabulary_
        embedding_matrix = self.train_svd(docterm_data)

        self.create_keyedvector_from_matrix(embedding_matrix, self.word2id)

    def train_svd(self, vectorized_corpus_data):
        """TODO
        Parameters
        ----------
        vectorized_corpus_data: TODO
            Vectorized data TODO
        """

        svd=TruncatedSVD(n_components=self.size)
        svd.fit(vectorized_corpus_data)

        embedding_matrix = svd.components_.T
        return embedding_matrix

    def train_word2vec(self):
        """TODO
        Parameters
        ----------
        """

        embedding = Word2Vec(workers=self.workers,
                             seed=self.seed,
                             iter=self.iter,
                             size=self.size,
                             window=self.window,
                             min_count=self.min_count)
        embedding.build_vocab(self.streamer.stream)
        embedding.train(self.streamer.stream,
                        total_examples=embedding.corpus_count,
                        epochs=embedding.epochs)

        self.embedding = embedding.wv

    #def train_glove(self, X, input_column):
    #
    #    corpus = Corpus()
    #
    #    corpus.fit(X[input_column], window=self.window)
    #
    #    self.word2id = corpus.dictionary
    #
    #    glove=Glove(no_components=self.size)
    #    glove.fit(corpus.matrix, epochs=self.iter, no_threads=self.workers, )
    #
    #    self.create_keyedvector_from_matrix(glove.word_vectors, self.word2id)

    def create_keyedvector_from_matrix(self, embedding_matrix, word2id):
        """TODO
        Parameters
        ----------
        embedding_matrix: numpy.ndarray
            Embedding matrix as a numpy object
        word2id: dict
            Word vocabulary (key: word, value: word_index)
        """

        vocab = word2id
        embedding_matrix = embedding_matrix
        vector_size = embedding_matrix.shape[1]

        kv = KeyedVectors(vector_size)
        kv.vector_size = vector_size
        kv.vectors = embedding_matrix

        kv.index2word = list(vocab.keys())

        kv.vocab = {word: Vocab(index=word_id, count=0) for word, word_id in vocab.items()}

        self.embedding = kv