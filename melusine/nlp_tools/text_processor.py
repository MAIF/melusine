import logging
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Dict, Sequence, Union
from abc import abstractmethod

from melusine.nlp_tools.base_melusine_class import BaseMelusineClass
from melusine.nlp_tools.lemmatizer import MelusineLemmatizer
from melusine.nlp_tools.normalizer import Normalizer, MelusineNormalizer
from melusine.nlp_tools.pipeline import MelusinePipeline
from melusine.nlp_tools.text_flagger import (
    MelusineTextFlagger,
    DeterministicTextFlagger,
)
from melusine.nlp_tools.token_flagger import MelusineTokenFlagger, FlashtextTokenFlagger
from melusine.nlp_tools.tokenizer import MelusineTokenizer, RegexTokenizer

logger = logging.getLogger(__name__)


class MelusineTextProcessor(BaseMelusineClass):
    CONFIG_KEY = "text_processor"
    FILENAME = "text_processor.json"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, text: str):
        raise NotImplementedError()


class TextProcessor(MelusineTextProcessor):
    """
    TextProcessor which does the following:
    - General flagging (using regex)
    - Join collocations (deterministic phrasing)
    - Name flagging (using FlashText)
    - Text splitting
    - Stopwords removal
    """

    def __init__(
        self,
        tokenizer: MelusineTokenizer,
        normalizer: MelusineNormalizer = None,
        form: str = None,
        lowercase: bool = None,
        tokenizer_regex: str = None,
        stopwords: Sequence[str] = None,
        text_flagger: MelusineTextFlagger = None,
        token_flagger: MelusineTokenFlagger = None,
        text_flags: Dict = None,
        token_flags: Dict = None,
        collocations: Dict = None,
        lemmatizer=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tokenizer_regex: str
            Regex used to split the text into tokens
        normalization: Union[str, None]
            Type of normalization to apply to the text
        lowercase: bool
            If True, lowercase the text
        stopwords: Sequence[str]
            List of words to be removed
        remove_stopwords: bool
            If True, stopwords removal is enabled
        flag_dict: Dict[str, str]
            Flagging dict with regex as key and replace_text as value
        collocations_dict: Dict[str, str]
            Dict with expressions to be grouped into one unit of sens
        """
        super().__init__()

        # Tokenizer
        if tokenizer_regex:
            if tokenizer:
                raise AttributeError(
                    f"You should specify only one of 'tokenizer_regex' and 'tokenizer'"
                )
            else:
                self.tokenizer = RegexTokenizer(
                    tokenizer_regex=tokenizer_regex, stopwords=stopwords
                )
        else:
            self.tokenizer = tokenizer

        # Normalizer
        if form:
            if form:
                raise AttributeError(
                    f"You should specify only one of 'form' and 'normalizer'"
                )
            else:
                self.normalizer = Normalizer(form=form, lowercase=lowercase)
        else:
            self.normalizer = normalizer

        # Text Flagger
        if text_flags:
            if text_flagger:
                raise AttributeError(
                    f"You should specify only one of 'text_flags' and 'text_flagger'"
                )
            else:
                self.text_flagger = DeterministicTextFlagger(text_flags=text_flags)
        else:
            self.text_flagger = text_flagger

        # Token Flagger
        if token_flags:
            if token_flagger:
                raise AttributeError(
                    f"You should specify only one of 'token_flags' and 'token_flagger'"
                )
            else:
                self.token_flagger = FlashtextTokenFlagger(token_flags=token_flags)
        else:
            self.token_flagger = token_flagger

        # Lemmatizer
        self.lemmatizer = lemmatizer

    def process(self, text: str) -> Sequence[str]:
        """
        Apply the full tokenization pipeline on a text.
        Parameters
        ----------
        text: str
            Input text to be tokenized
        Returns
        -------
        tokens: Sequence[str]
            List of tokens
        """
        # Normalize text
        text = self.normalizer.normalize(text)

        # Text flagging
        if self.text_flagger is not None:
            text = self.text_flagger.flag_text(text)

        # Text splitting
        tokens = self.tokenizer.tokenize(text)

        # Lemmatizer
        if self.lemmatizer is not None:
            tokens = self.lemmatizer.lemmatize(tokens)

        # Token Flagging
        if self.token_flagger is not None:
            tokens = self.token_flagger.flag_tokens(tokens)

        return tokens

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the TextProcessor into a json file
        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            A prefix to add to the names of the files saved by the tokenizer.
        """
        d = self.__dict__.copy()

        # Save Phraser
        self.save_obj(d, path, filename_prefix, MelusineNormalizer.CONFIG_KEY)

        # Save Text Flagger
        self.save_obj(d, path, filename_prefix, MelusineTextFlagger.CONFIG_KEY)

        # Save Lemmatizer
        self.save_obj(d, path, filename_prefix, MelusineLemmatizer.CONFIG_KEY)

        # Save Token Flagger
        self.save_obj(d, path, filename_prefix, MelusineTokenFlagger.CONFIG_KEY)

        # Save Tokenizer
        self.save_obj(d, path, filename_prefix, MelusineTokenizer.CONFIG_KEY)

        # Save Tokenizer
        self.save_json(
            save_dict=d,
            path=path,
            filename_prefix=filename_prefix,
        )

    @classmethod
    def load(cls, path: str):
        """
        Load the tokenizer from a json file
        Parameters
        ----------
        path: str
            Load path
        Returns
        -------
        _: TextProcessor
            TextProcessor instance
        """
        # Load the Tokenizer config file
        config_dict = cls.load_json(path)

        # Load Normalizer
        normalizer = cls.load_obj(
            config_dict, path=path, obj_key=MelusineNormalizer.CONFIG_KEY
        )

        # Load Text Flagger
        text_flagger = cls.load_obj(
            config_dict, path=path, obj_key=MelusineTextFlagger.CONFIG_KEY
        )

        # Load lemmatizer
        lemmatizer = cls.load_obj(
            config_dict, path=path, obj_key=MelusineLemmatizer.CONFIG_KEY
        )

        # Load token flagger
        token_flagger = cls.load_obj(
            config_dict, path=path, obj_key=MelusineTokenFlagger.CONFIG_KEY
        )

        # Load Tokenizer
        tokenizer = cls.load_obj(
            config_dict, path=path, obj_key=MelusineTokenizer.CONFIG_KEY
        )

        return cls(
            token_flagger=token_flagger,
            text_flagger=text_flagger,
            lemmatizer=lemmatizer,
            tokenizer=tokenizer,
            normalizer=normalizer,
            **config_dict,
        )


def create_pipeline(
    form: str = None,
    lowercase: bool = None,
    tokenizer_regex: str = None,
    stopwords: Sequence[str] = None,
    text_flags: Dict = None,
    token_flags: Dict = None,
    memory=False,
    verbose=False,
):

    steps = list()

    # Normalizer
    if form:
        normalizer = Normalizer(form=form, lowercase=lowercase)
        steps.append(("normalizer", normalizer))

    # Text Flagger
    if text_flags:
        text_flagger = DeterministicTextFlagger(text_flags=text_flags)
        steps.append(("text_flagger", text_flagger))

        # Tokenizer
    if not tokenizer_regex:
        raise AttributeError("Tokenizer regexp is mandatory")
    else:
        tokenizer = RegexTokenizer(tokenizer_regex=tokenizer_regex, stopwords=stopwords)
        steps.append(("tokenizer", tokenizer))

    # Token Flagger
    if token_flags:
        token_flagger = FlashtextTokenFlagger(token_flags=token_flags)
        steps.append(("token_flagger", token_flagger))

    return MelusinePipeline(steps=steps, memory=memory, verbose=verbose)
