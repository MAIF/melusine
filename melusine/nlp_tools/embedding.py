import logging
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from melusine.nlp_tools.tokenizer import WordLevelTokenizer


logger = logging.getLogger(__name__)


class Word2VecTrainer:
    def __init__(
        self, input_column=None, tokens_column="tokens", tokenizer=None, **kwargs
    ):

        self.input_column = input_column
        self.tokens_column = tokens_column

        if tokenizer is None:
            self.tokenizer = WordLevelTokenizer()
        else:
            self.tokenizer = tokenizer

        self.model_parameters_dict = kwargs
        self.embedding = None

    def train(self, x):
        """Fits a Word2Vec Embedding on the given documents, and update the embedding attribute."""

        if not self.tokens_column in x.columns:
            x[self.tokens_column] = x[self.input_column].apply(self.tokenizer.tokenize)

        embedding = Word2Vec(**self.model_parameters_dict)

        embedding.build_vocab(x[self.tokens_column])
        embedding.train(
            x[self.tokens_column],
            total_examples=embedding.corpus_count,
            epochs=embedding.epochs,
        )

        self.embedding = embedding.wv


class Embedding:
    """
    This class is deprecated and should not be used.
    Is is kept to ensure retro-compatibility with earlier Melusine versions.
    It will be removed eventually.

    """

    def __init__(
        self,
        input_column=None,
        tokens_column=None,
        method="word2vec_cbow",
        stop_removal=True,
        **kwargs
    ):
        self.input_column = input_column
        self.tokens_column = tokens_column
        self.method = method
        self.stop_removal = stop_removal
        self.train_params = kwargs

        self.tokenizer = WordLevelTokenizer(remove_stopwords=self.stop_removal)
        self.trainer = Word2VecTrainer()
        self.embedding = None

    def train(self, x):
        # Instantiate the trainer
        embedding_trainer = Word2VecTrainer(
            input_column=self.input_column,
            tokens_column=self.tokens_column,
            tokenizer=self.tokenizer,
            **self.train_params
        )

        # Train the word embeddings model
        embedding_trainer.train(x)

        self.embedding = embedding_trainer.embedding

    def save(self, path):
        self.embedding.save(path)

    @classmethod
    def load(cls, path):
        embedding = KeyedVectors.load(path)

        # input_column is useless as the model should be trained already
        instance = cls(input_column="body")
        instance.embedding = embedding

        return instance
