import logging
import gensim
import pickle
import re
from melusine import config
from melusine.nlp_tools.tokenizer import WordLevelTokenizer

_common_terms = config["words_list"]["stopwords"] + config["words_list"]["names"]

regex_tokenize_with_punctuations = r"(.*?[\s'])"
tokenize_without_punctuations = r"(.*?)[\s']"
regex_process = r"\w+(?:[\?\-'\"_]\w+)*"
regex_split_parts = r"(.*?[;.,?!])"

logger = logging.getLogger(__name__)


def phraser_on_body(row, phraser):
    """Applies phraser on cleaned body.

    To be used with methods such as: `apply(func, axis=1)` or
    `apply_by_multiprocessing(func, axis=1, **kwargs)`.

    Parameters
    ----------
    row : row of pd.Dataframe

    phraser : Phraser instance,

    Returns
    -------
    pd.Series

    Examples
    --------
        >>> import pandas as pd
        >>> from melusine.nlp_tools.phraser import phraser_on_body
        >>> from melusine.nlp_tools.phraser import Phraser
        >>> from melusine import load_email_data
        >>> data = load_email_data()
        >>> # data contains a 'clean_body' column
        >>> phraser = Phraser(columns='clean_body').load(filepath)
        >>> data.apply(phraser_on_body, axis=1)  # apply to all samples

    """
    clean_body = phraser_on_text(row["clean_body"], phraser)

    return clean_body


def phraser_on_header(row, phraser):
    """Applies phraser on cleaned header.

    To be used with methods such as: `apply(func, axis=1)` or
    `apply_by_multiprocessing(func, axis=1, **kwargs)`.

    Parameters
    ----------
    row : row of pd.Dataframe

    phraser : Phraser instance,

    Returns
    -------
    pd.Series

    Examples
    --------
        >>> import pandas as pd
        >>> from melusine.nlp_tools.phraser import phraser_on_header
        >>> from melusine.nlp_tools.phraser import Phraser
        >>> from melusine import load_email_data
        >>> data = load_email_data()
        >>> # data contains a 'clean_body' column
        >>> phraser = Phraser(columns='clean_header').load(filepath)
        >>> data.apply(phraser_on_header, axis=1)  # apply to all samples

    """
    clean_header = phraser_on_text(row["clean_header"], phraser)

    return clean_header


def phraser_on_text(text, phraser):
    """Returns text with phrased words.

    Parameters
    ----------
    text : str,

    phraser : Phraser instance,

    Returns
    -------
    str

    """
    if not re.search(pattern=r"\W*\b\w+\b\W*", string=text):
        return text
    (
        pre_typos_list,
        words_list,
        separators_list,
    ) = _split_typos_words_separators(text)
    phrased_words_list = phraser.phraser[words_list]
    phrased_text = _rebuild_phrased_text_with_punctuation(
        pre_typos_list, words_list, separators_list, phrased_words_list
    )

    return phrased_text


def _rebuild_phrased_text_with_punctuation(
    pre_typos_list, words_list, separators_list, phrased_words_list
):
    """Rebuilds the initial text with phrased words."""
    i = 0
    for pre_typo, word, separator in zip(pre_typos_list, words_list, separators_list):
        phrased_word = re.sub(r"\W", "", phrased_words_list[i])
        word = re.sub(r"\W", "", word)
        if len(phrased_word) > len(word):
            if _check_last_word_phrased(phrased_word, word):
                phrased_words_list[i] = pre_typo + phrased_word + separator
                i += 1
        else:
            phrased_words_list[i] = pre_typo + phrased_word + separator
            i += 1

    return "".join(phrased_words_list)


def _check_last_word_phrased(phrased_word, word):
    """Check if a word is the last word of a phrased word."""
    words_list = phrased_word.split("_")
    last_word = words_list[-1]

    return word == last_word


def _split_typos_words_separators(text, pattern=r"(\W*)\b(\w+)\b(\W*)"):
    """Split text according to typos."""
    tuple_word_separator_list = re.findall(pattern, text, flags=re.M | re.I)
    pre_typos_list, words_list, separators_list = zip(*tuple_word_separator_list)

    return pre_typos_list, words_list, separators_list


class Phraser:
    """Class to train a phraser.

    Parameters
    ----------
    input_column : str,
        Input text column to consider for the phraser.

    common_terms : list of integers, optional
        List of stopwords.
        Default value, list of stopwords from nltk.

    threshold : int, optional
        Threshold to select colocations.
        Default value, 350.

    min_count : int, optional
        Minimum count of word to be selected as colocation.
        Default value, 200.

    Attributes
    ----------
    common_terms, threshold, min_count,

    phraser : Phraser object from Gensim

    Examples
    --------
    >>> from melusine.nlp_tools.phraser import Phraser
    >>> phraser = Phraser()
    >>> phraser.train(X)
    >>> phraser.save(filepath)
    >>> phraser = phraser().load(filepath)

    """

    def __init__(
        self,
        input_column="clean_body",
        common_terms=_common_terms,
        threshold=350,
        min_count=200,
        tokenizer=None,
    ):
        self.common_terms = common_terms
        self.threshold = threshold
        self.min_count = min_count
        self.input_column = input_column
        self.phraser = None

        if tokenizer is None:
            self.tokenizer = WordLevelTokenizer()
        else:
            self.tokenizer = tokenizer

    def save(self, filepath):
        """Method to save Phraser object"""
        with open(filepath, "wb") as f:
            pickle.dump(self.phraser, f)

    def load(self, filepath):
        """Method to load Phraser object"""
        with open(filepath, "rb") as f:
            self.phraser = pickle.load(f)
        return self

    def train(self, X):
        """Train phraser.

        Parameters
        ----------
        X : pd.Dataframe

        Returns
        -------
        self : object
            Returns the instance
        """
        logger.info("Start training for colocation detector")
        input_data = X[self.input_column].apply(self.tokenizer.tokenize)

        if gensim.__version__.startswith("3"):
            phrases = gensim.models.Phrases(
                input_data,
                common_terms=self.common_terms,
                threshold=self.threshold,
                min_count=self.min_count,
            )
        else:
            phrases = gensim.models.Phrases(
                input_data,
                connector_words=self.common_terms,
                threshold=self.threshold,
                min_count=self.min_count,
            )

        self.phraser = gensim.models.phrases.Phraser(phrases)
        logger.info("Done.")
