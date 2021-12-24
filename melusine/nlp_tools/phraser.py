import logging
import gensim

logger = logging.getLogger(__name__)


class Phraser:
    """
    Train and use a Gensim Phraser.
    """

    FILENAME = "gensim_phraser_meta.pkl"
    PHRASER_FILENAME = "gensim_phraser"

    def __init__(self, input_column="tokens", output_column="tokens", **phraser_args):

        self.input_column = input_column
        self.output_column = output_column
        self.phraser_args = phraser_args
        self.phraser_ = None

    def fit(self, df, y=None):
        """ """
        input_data = df[self.input_column]
        phrases = gensim.models.Phrases(input_data, **self.phraser_args)
        self.phraser_ = gensim.models.phrases.Phraser(phrases)

        return self

    def transform(self, df):
        df[self.output_column] = self.phraser_[df[self.input_column]]
        return df
