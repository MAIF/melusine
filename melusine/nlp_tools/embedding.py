import logging
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Vocab, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
# from glove import Corpus, Glove

from melusine.utils.streamer import Streamer

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%d/%m %I:%M')


class Embedding:
    """Class to train embeddings with Word2Vec algorithm.
    Attributes
    ----------
    input_column : str,
        String of the input column name for the pandas dataframe to compute the embedding on.
    streamer : Streamer instance,
        Builds a stream a tokens from a pd.Dataframe to train the embeddings.
    word2id: dict,
        Word vocabulary (key: word, value: word_index.
    embedding : Gensim KeyedVector Instance,
        Gensim KeyedVector Instance relative to the specific trained or imported embedding.
    method : str,
        One of the following :
            - "word2vec_sg" : Trains a Word2Vec Embedding using the Skip-Gram method (usually takes a long time).
            - "word2vec_ns" : Trains a Word2Vec Embedding using the Negative-Sampling method.
            - "word2vec_cbow" : Trains a Word2Vec Embedding using the Continuous Bag-Of-Words method.
            - "lsa_docterm" : Trains an Embedding by using an SVD on a Document-Term Matrix.
            - "lsa_tfidf" : Trains an Embedding by using an SVD on a TF-IDFized Document-Term Matrix.
            - "glove" : Trains a GloVe Embedding. NOT IMPLEMENTED YET.
    train_params : dict,
        Additional parameters for the embedding training. Check the following documentation :
            - gensim.models.Word2Vec for Word2Vec Embeddings
            - sklearn.decomposition.TruncatedSVD for LSA Embeddings
            - glove.Glove for GloVe Embeddings
        If left untouched, the default training values will be kept from the aforementioned packages.
    Examples
    --------
    >>> from melusine.nlp_tools.embedding import Embedding
    >>> embedding = Embedding()
    >>> embedding.train(X)
    >>> embedding.save(filepath)
    >>> embedding = Embedding().load(filepath)
    """

    def __init__(self,
                 input_column=None,
                 tokens_column=None,
                 workers=40,
                 random_seed=42,
                 iter=15,
                 size=300,
                 method="word2vec_cbow",
                 stop_removal=True,
                 min_count=100
                 ):
        """
        Parameters :
        ----------
        input_column : str,
            String of the input column name for the pandas dataframe to compute the embedding on (default="clean_text").
        workers : int,
            Number of CPUs to use for the embedding training process (default=40).
        random_seed : int,
            Seed for reproducibility (default=42).
        iter : int,
            Number of epochs (default=15). Used for Word2Vec and GloVe only.
        size : int,
            Desired embedding size (default=300).
        window : int,
            If Word2Vec, window used to find center-context word relationships.
            If GloVe, window used to compute the co-occurence matrix.
        min_count : int,
            Minimum number of appeareance of a token in the corpus for it to be kept in the vocabulary (default=100).
        stop_removal : bool,
            If True, removes stopwords in the Streamer process (default=True).
        method : str,
            One of the following :
                - "word2vec_sg" : Trains a Word2Vec Embedding using the Skip-Gram method and Negative-Sampling.
                - "word2vec_cbow" : Trains a Word2Vec Embedding using the Continuous Bag-Of-Words method and Negative-Sampling.
                - "lsa_docterm" : Trains an Embedding by using an SVD on a Document-Term Matrix.
                - "lsa_tfidf" : Trains an Embedding by using an SVD on a TF-IDFized Document-Term Matrix.
                - "glove" : Trains a GloVe Embedding. NOT IMPLEMENTED YET.
        min_count : TODO
        """

        self.logger = logging.getLogger(__name__)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.debug('Create an Embedding instance.')
        self.input_column = input_column
        self.tokens_column = tokens_column
        self.input_data = None
        self.streamer = Streamer(column=self.input_column, stop_removal=stop_removal)
        self.word2id = {}
        self.embedding = None
        self.method = method
        self.workers = workers

        if self.method in ['word2vec_sg', 'word2vec_cbow']:
            self.train_params = {
                 "size": size,
                 "alpha": 0.025,
                 "min_count": min_count,
                 "max_vocab_size": None,
                 "sample": 0.001,
                 "seed": random_seed,
                 "workers": workers,
                 "min_alpha": 0.0001,
                 "negative" : 5,
                 "hs": 0,
                 "ns_exponent": 0.75,
                 "cbow_mean": 1,
                 "iter": iter,
                 "null_word": 0,
                 "trim_rule": None,
                 "sorted_vocab": 1,
                 "batch_words": 10000,
                 "compute_loss": False,
                 "callbacks": (),
                 "max_final_vocab": None
                 }
            if self.method == "word2vec_sg":
                self.train_params["sg"] = 1
                self.train_params["window"] = 10

            elif self.method == "word2vec_cbow":
                self.train_params["sg"] = 0
                self.train_params["window"] = 5

        elif self.method in 'lsa_tfidf':
            self.train_params = {
                # TfidfVectorizer Parameters
                "vectorizer_encoding": 'utf-8',
                "vectorizer_decode_error": 'strict',
                "vectorizer_strip_accents": None,
                "vectorizer_lowercase": False,
                "vectorizer_analyzer": 'word',
                "vectorizer_stop_words": None,
                "vectorizer_ngram_range": (1, 1),
                "vectorizer_max_df": 1.0,
                "vectorizer_min_df": 1,
                "vectorizer_max_features": None,
                "vectorizer_vocabulary": None,
                "vectorizer_binary": False,
                "vectorizer_norm": 'l2',
                "vectorizer_use_idf": True,
                "vectorizer_smooth_idf": True,
                "vectorizer_sublinear_tf": False,

                # TruncatedSVD Parameters
                "svd_n_components": size,
                "svd_algorithm": 'randomized',
                "svd_n_iter": iter,
                "svd_random_state": random_seed,
                "svd_tol": 0.0
            }
        elif self.method == 'lsa_docterm':
            self.train_params = {
                # CountVectorizer Parameters
                "vectorizer_encoding": 'utf-8',
                "vectorizer_decode_error": 'strict',
                "vectorizer_strip_accents": None,
                "vectorizer_lowercase": False,
                "vectorizer_stop_words": None,
                "vectorizer_ngram_range": (1, 1),
                "vectorizer_analyzer": 'word',
                "vectorizer_max_df": 1.0,
                "vectorizer_min_df": 1,
                "vectorizer_max_features": None,
                "vectorizer_vocabulary": None,
                "vectorizer_binary": False,

                # TruncatedSVD Parameters
                "svd_n_components": size,
                "svd_algorithm": 'randomized',
                "svd_n_iter": iter,
                "svd_random_state": random_seed,
                "svd_tol": 0.0
                 }
        else:
            raise ValueError(f"Error: Embedding method {method} not recognized or not implemented yet ;).")

    def save(self, filepath):
        """Method to save Embedding object."""
        self.embedding.save(filepath)

    def load(self, filepath):
        """Method to load Embedding object."""
        self.embedding = KeyedVectors.load(filepath, mmap='r')
        return self

    def train(self, X):
        """Train embeddings with the desired word embedding algorithm (default is Word2Vec).
        Parameters
        ----------
        X : pd.Dataframe
            Containing a clean body column.
        """

        self.logger.info('Start training for embedding')

        if (not self.input_column) and (not self.tokens_column):
            raise ValueError("""
            Please specify one of the following keyword argument when instanciating an Embedding class object: 
            - input_column argument (containing raw text) 
            - tokens_column argument (containing) list of tokens""")

        # If tokens column is provided: use it. Otherwise, use input column
        if self.tokens_column:
            self.input_data = X[self.tokens_column].tolist()
        elif self.input_column:
            self.streamer.to_stream(X)
            self.input_data = self.streamer.stream
        else:
            raise ValueError("""
            Please specify one of the following keyword argument when instanciating an Embedding class object: 
            - input_column argument (containing raw text) 
            - tokens_column argument (containing) list of tokens""")

        if self.method in ['word2vec_sg', 'word2vec_cbow']:
            self.train_word2vec()
        elif self.method == 'lsa_tfidf':
            self.train_tfidf()
        elif self.method == 'lsa_docterm':
            self.train_docterm()
        # elif embedding_type == 'glove':
        #    self.train_glove(X, self.input_column)

        self.logger.info('Done.')

    def train_tfidf(self):
        """Train a TF-IDF Vectorizer to compute a TF-IDFized Doc-Term Matrix relative to a corpus.

        Parameters
        ----------
        """

        def dummy_function(doc):
            return doc

        tfidf_vec = TfidfVectorizer(
            tokenizer=dummy_function,
            preprocessor=dummy_function,
            token_pattern=None,

            encoding=self.train_params["vectorizer_encoding"],
            decode_error=self.train_params["vectorizer_decode_error"],
            strip_accents=self.train_params["vectorizer_strip_accents"],
            lowercase=self.train_params["vectorizer_lowercase"],
            analyzer=self.train_params["vectorizer_analyzer"],
            stop_words=self.train_params["vectorizer_stop_words"],
            ngram_range=self.train_params["vectorizer_ngram_range"],
            max_df=self.train_params["vectorizer_max_df"],
            min_df=self.train_params["vectorizer_min_df"],
            max_features=self.train_params["vectorizer_max_features"],
            vocabulary=self.train_params["vectorizer_vocabulary"],
            binary=self.train_params["vectorizer_binary"],
            use_idf=self.train_params["vectorizer_use_idf"],
            smooth_idf=self.train_params["vectorizer_smooth_idf"],
            sublinear_tf=self.train_params["vectorizer_sublinear_tf"]
        )

        tfidf_data = tfidf_vec.fit_transform(self.input_data)

        self.word2id = tfidf_vec.vocabulary_
        embedding_matrix = self.train_svd(tfidf_data)

        self.create_keyedvector_from_matrix(embedding_matrix, self.word2id)

    def train_docterm(self):
        """Train a Count Vectorizer to compute a Doc-Term Matrix relative to a corpus.
        Parameters
        ----------
        X : pd.Dataframe
            Containing a column with tokenized documents
        input_column: str
            Name of the input column containing tokens
        """

        def dummy_function(doc):
            return doc

        # CountVectorizer
        count_vec = CountVectorizer(
            tokenizer=dummy_function,
            preprocessor=dummy_function,
            token_pattern=None,

            encoding=self.train_params["vectorizer_encoding"],
            decode_error=self.train_params["vectorizer_decode_error"],
            strip_accents=self.train_params["vectorizer_strip_accents"],
            lowercase=self.train_params["vectorizer_lowercase"],
            stop_words=self.train_params["vectorizer_stop_words"],
            ngram_range=self.train_params["vectorizer_ngram_range"],
            analyzer=self.train_params["vectorizer_analyzer"],
            max_df=self.train_params["vectorizer_max_df"],
            min_df=self.train_params["vectorizer_min_df"],
            max_features=self.train_params["vectorizer_max_features"],
            vocabulary=self.train_params["vectorizer_vocabulary"],
            binary=self.train_params["vectorizer_binary"]
        )

        docterm_data = count_vec.fit_transform(self.input_data)

        self.word2id = count_vec.vocabulary_
        embedding_matrix = self.train_svd(docterm_data)

        self.create_keyedvector_from_matrix(embedding_matrix, self.word2id)

    def train_svd(self, vectorized_corpus_data):
        """Fits a TruncatedSVD on a Doc-Term/TF-IDFized Doc-Term Matrix for dimensionality reduction.
        Parameters
        ----------
        vectorized_corpus_data: CountVectorizer or TfidfVectorizer object,
            Sklearn object on which the TruncatedSVD will be computed.
        """

        svd=TruncatedSVD(n_components=self.train_params["svd_n_components"],
                        algorithm=self.train_params["svd_algorithm"],
                        n_iter=self.train_params["svd_n_iter"],
                        random_state=self.train_params["svd_random_state"],
                        tol=self.train_params["svd_tol"])
        svd.fit(vectorized_corpus_data)

        embedding_matrix = svd.components_.T
        return embedding_matrix

    def train_word2vec(self):
        """Fits a Word2Vec Embedding on the given documents, and update the embedding attribute.
        """


        embedding = Word2Vec(size=self.train_params["size"],
                             alpha=self.train_params["alpha"],
                             window=self.train_params["window"],
                             min_count=self.train_params["min_count"],
                             max_vocab_size=self.train_params["max_vocab_size"],
                             sample=self.train_params["sample"],
                             seed=self.train_params["seed"],
                             workers=self.train_params["workers"],
                             min_alpha=self.train_params["min_alpha"],
                             sg=self.train_params["sg"],
                             hs=self.train_params["hs"],
                             negative=self.train_params["negative"],
                             ns_exponent=self.train_params["ns_exponent"],
                             cbow_mean=self.train_params["cbow_mean"],
                             iter=self.train_params["iter"],
                             null_word=self.train_params["null_word"],
                             trim_rule=self.train_params["trim_rule"],
                             sorted_vocab=self.train_params["sorted_vocab"],
                             batch_words=self.train_params["batch_words"],
                             compute_loss=self.train_params["compute_loss"],
                             callbacks=self.train_params["callbacks"],
                             max_final_vocab=self.train_params["max_final_vocab"])

        # TODO Fix Streamer
        embedding.build_vocab(self.input_data)
        embedding.train(self.input_data,
                        total_examples=embedding.corpus_count,
                        epochs=self.train_params["iter"])

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
        """
        Imports the necessary attributes for the Embedding object from an embedding matrix and a word2id vocabulary. Can be used for custom pre-trained embeddings.
        Parameters
        ----------
        embedding_matrix: numpy.ndarray
            Embedding matrix as a numpy object
        word2id: dict
            Word vocabulary (key: word, value: word_index)
        """

        vocab = {word:word2id[word] for word in sorted(word2id, key=word2id.__getitem__, reverse=False)}
        embedding_matrix = embedding_matrix
        vector_size = embedding_matrix.shape[1]

        kv = KeyedVectors(vector_size)
        kv.vector_size = vector_size
        kv.vectors = embedding_matrix

        kv.index2word = list(vocab.keys())

        kv.vocab = {word: Vocab(index=word_id, count=0) for word, word_id in vocab.items()}

        self.embedding = kv
