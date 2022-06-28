import logging
from nltk.stem import SnowballStemmer

logger = logging.getLogger(__name__)


class Stemmer:
    """Compute list Series which return the stemmed version of a list of tokens
    Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words.

    Parameters
    ----------
    input_column : str,
        Column of pd.Dataframe which contains a list of tokens, default column ['tokens']
    output_column: str,
        Column where is saved the list of stemmed tokens, default column ['stemmed_tokens']
    language : str,
        Language of the tokens to be stemmed.
        Supported languages : 'arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'norwegian', 'porter', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish' 
        Default value, 'french'
    
    Returns
    -------
    pd.Dataframe
    Examples
    --------
        >>> from melusine.prepare_email.cleaning import Stemmer
        >>> stemmer = Stemmer()
        >>> stemmer.transform(data)
    """

    FILENAME = "nltk_stemmer_meta.pkl"
    STEMMER_FILENAME = "nltk_stemmer"

    def __init__(self, input_column: str ="tokens", output_column: str ="stemmed_tokens", language: str = 'french'):

        self.input_column = input_column
        self.output_column = output_column
        self.stemmer = SnowballStemmer(language)
    
    def _stemming(self, input_tokens: list):
        return [self.stemmer.stem(token) for token in input_tokens]


    def fit(self, df, y=None):
        """ """
        return self

    def transform(self, df):
        input_data = df[self.input_column]
        df[self.output_column] = input_data.apply(self._stemming)
        return df