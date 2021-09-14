import logging
from gensim.models import Word2Vec
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
