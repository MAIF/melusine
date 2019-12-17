import logging
import re
from sklearn.base import BaseEstimator, TransformerMixin
from melusine.config.config import ConfigJsonReader
from melusine.utils.transformer_scheduler import TransformerScheduler

conf_reader = ConfigJsonReader()
config = conf_reader.get_config_file()

stopwords = config["words_list"]["stopwords"]
names_list = config["words_list"]["names"]
regex_tokenize = config["regex"]["tokenizer"]


class Tokenizer(BaseEstimator, TransformerMixin):
    """Class to train and apply tokenizer.

    Compatible with scikit-learn API (i.e. contains fit, transform methods).

    Parameters
    ----------
    input_column : str,
        Input text column to consider for the tokenizer.

    stopwords : list of strings, optional
        List of words to remove from list of tokens.
        Default value, list defined in conf file

    stop_removal : boolean, optional
        True if stopwords to be removed, else False.
        Default value, False.

    n_jobs : int, optional
        Number of cores used for computation.
        Default value, 20.

    Attributes
    ----------
    stopwords, stop_removal, n_jobs

    Examples
    --------
    >>> from melusine.nlp_tools.tokenizer import Tokenizer
    >>> tokenizer = Tokenizer()
    >>> X = tokenizer.fit_transform(X)
    >>> tokenizer.save(filepath)
    >>> tokenizer = Tokenizer().load(filepath)

    """

    def __init__(self,
                 input_column='clean_text',
                 stopwords=stopwords,
                 stop_removal=True,
                 n_jobs=20):
        self.input_column = input_column
        self.stopwords = set(stopwords)
        self.stop_removal = stop_removal
        self.names_list = set(names_list)
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)

    def __getstate__(self):
        """should return a dict of attributes that will be pickled
        To override the default pickling behavior and
        avoid the pickling of the logger
        """
        d = self.__dict__.copy()
        d['n_jobs'] = 1
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        """To override the default pickling behavior and
        avoid the pickling of the logger"""
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

    def fit(self, X, y=None):
        """Unused method. Defined only for compatibility with scikit-learn API.
        """
        return self

    def transform(self, X):
        """Applies tokenize method on pd.Dataframe.

        Parameters
        ----------
        X : pandas.DataFrame,
            Data on which transformations are applied.

        Returns
        -------
        pandas.DataFrame
        """
        self.logger.debug('Start transform tokenizing')

        if isinstance(X, dict):
            apply_func = TransformerScheduler.apply_dict
        else:
            apply_func = TransformerScheduler.apply_pandas

        X['tokens'] = apply_func(X, self.tokenize)
        X['tokens'] = apply_func(X, lambda x: x['tokens'][0])

        self.logger.debug('Done.')
        return X

    def tokenize(self, row):
        """Returns list of tokens.

        Parameters
        ----------
        row : row of pd.Dataframe

        Returns
        -------
        pd.Series

        """
        text = row[self.input_column]
        tokens = self._tokenize(text)
        return [tokens]

    def _tokenize(self, text, pattern=regex_tokenize):
        """Returns list of tokens from text."""
        if isinstance(text, str):
            tokens = re.findall(pattern, text, re.M+re.DOTALL)
            tokens = self._remove_stopwords(tokens)
        else:
            tokens = []
        return tokens

    def _remove_stopwords(self, list):
        """ Removes stopwords from list if stop_removal parameter
        set to True."""
        if self.stop_removal:
            return [tok if tok not in self.names_list else "flag_name_" for tok in list if tok not in self.stopwords]
        else:
            return list
