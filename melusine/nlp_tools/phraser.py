import logging
import gensim
import re
from melusine import config
from melusine.core.melusine_transformer import MelusineTransformer

_common_terms = config["tokenizer"]["stopwords"] + config["tokenizer"]["names"]

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
        >>> data = load_email_data(type="preprocessed")
        >>> # data contains a 'clean_body' column
        >>> phraser = Phraser.load(filepath)
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


class Phraser(MelusineTransformer):
    """ """

    FILENAME = "gensim_phraser.pkl"

    def __init__(self, input_columns, output_columns, **phraser_args):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=None,
        )
        self.phraser_args = phraser_args
        self.phraser_ = None

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the Token Flagger into a pkl file

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            Prefix for saved files.
        """
        self.save_pkl(path, filename_prefix)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load the DeterministicTextFlagger from a json file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str
        Returns
        -------
        _: Phraser
            Phraser instance
        """
        return cls.load_pkl(path, filename_prefix=filename_prefix)

    def fit(self, df, y=None):
        """ """
        input_data = df[self.input_columns[0]]
        phrases = gensim.models.Phrases(input_data, **self.phraser_args)
        self.phraser_ = gensim.models.phrases.Phraser(phrases)

        return self

    def transform(self, df):
        df[self.input_columns[0]] = self.phraser_[df[self.output_columns[0]]]
        return df
