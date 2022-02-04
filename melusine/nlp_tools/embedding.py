import logging
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Vocab, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


logger = logging.getLogger(__name__)


class Embedding:
    """Class to train embeddings with Word2Vec algorithm.
    Attributes
    ----------
    word2id: dict,
        Word vocabulary (key: word, value: word_index.
    embedding : Gensim KeyedVector Instance,
        Gensim KeyedVector Instance relative to the specific trained or imported embedding.
    method : str,
        One of the following :
            - "word2vec_sg" : Trains a Word2Vec Embedding using the Skip-Gram method (usually takes a long time).
            - "word2vec_cbow" : Trains a Word2Vec Embedding using the Continuous Bag-Of-Words method.
            - "lsa_docterm" : Trains an Embedding by using an SVD on a Document-Term Matrix.
            - "lsa_tfidf" : Trains an Embedding by using an SVD on a TF-IDFized Document-Term Matrix.
    train_params : dict,
        Additional parameters for the embedding training. Check the following documentation :
            - gensim.models.Word2Vec for Word2Vec Embeddings
            - sklearn.decomposition.TruncatedSVD for LSA Embeddings
        If left untouched, the default training values will be kept from the aforementioned packages.
    Examples
    --------
    >>> from melusine.nlp_tools.embedding import Embedding
    >>> embedding = Embedding()
    >>> embedding.train(X)  # noqa
    >>> embedding.save(filepath)  # noqa
    >>> embedding = Embedding().load(filepath)  # noqa
    """

    def __init__(
        self,
        tokens_column=None,
        workers=40,
        random_seed=42,
        iter=15,
        size=300,
        method="word2vec_cbow",
        min_count=100,
    ):
        """
        Parameters :
        ----------
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
                - "word2vec_cbow" : Trains a Word2Vec Embedding using the Continuous Bag-Of-Words method
                                    and Negative-Sampling.
                - "lsa_docterm" : Trains an Embedding by using an SVD on a Document-Term Matrix.
                - "lsa_tfidf" : Trains an Embedding by using an SVD on a TF-IDFized Document-Term Matrix.
        min_count : int
            Minimum number of occurence of a word to be included in the vocabulary
        """
        self.tokens_column = tokens_column
        self.input_data = None
        self.word2id = {}
        self.embedding = None
        self.method = method
        self.workers = workers

        if self.method in ["word2vec_sg", "word2vec_cbow"]:
            self.train_params = {
                "size": size,
                "alpha": 0.025,
                "min_count": min_count,
                "max_vocab_size": None,
                "sample": 0.001,
                "seed": random_seed,
                "workers": workers,
                "min_alpha": 0.0001,
                "negative": 5,
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
                "max_final_vocab": None,
            }

            if self.method == "word2vec_sg":
                self.train_params["sg"] = 1
                self.train_params["window"] = 10

            elif self.method == "word2vec_cbow":
                self.train_params["sg"] = 0
                self.train_params["window"] = 5

        else:
            raise ValueError(
                f"Error: Embedding method {method} not recognized or not implemented yet ;)."
            )

    def save(self, filepath):
        """Method to save Embedding object."""
        self.embedding.save(filepath)

    def load(self, filepath):
        """Method to load Embedding object."""
        self.embedding = KeyedVectors.load(filepath, mmap="r")
        return self

    def train(self, X):
        """Train embeddings with the desired word embedding algorithm (default is Word2Vec).
        Parameters
        ----------
        X : pd.Dataframe
            Containing a clean body column.
        """

        logger.info("Start training word embeddings")

        # Get token corpus
        self.input_data = X[self.tokens_column].tolist()

        if self.method in ["word2vec_sg", "word2vec_cbow"]:
            self.train_word2vec()

        logger.info("Finished training word embeddings")

    def train_word2vec(self):
        """Fits a Word2Vec Embedding on the given documents, and update the embedding attribute."""
        embedding = Word2Vec(
            vector_size=self.train_params["size"],
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
            epochs=self.train_params["iter"],
            null_word=self.train_params["null_word"],
            trim_rule=self.train_params["trim_rule"],
            sorted_vocab=self.train_params["sorted_vocab"],
            batch_words=self.train_params["batch_words"],
            compute_loss=self.train_params["compute_loss"],
            callbacks=self.train_params["callbacks"],
            max_final_vocab=self.train_params["max_final_vocab"],
        )

        embedding.build_vocab(self.input_data)
        embedding.train(
            self.input_data,
            total_examples=embedding.corpus_count,
            epochs=self.train_params["iter"],
        )

        self.embedding = embedding.wv
