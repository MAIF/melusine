import logging

from typing import Dict, Sequence, Union

from melusine.nlp_tools.normalizer import Normalizer
from melusine.core.pipeline import MelusinePipeline
from melusine.nlp_tools.text_flagger import (
    DeterministicTextFlagger,
)
from melusine.nlp_tools.token_flagger import FlashtextTokenFlagger
from melusine.nlp_tools.tokenizer import RegexTokenizer

logger = logging.getLogger(__name__)


def make_tokenizer(
    form: str = None,
    lowercase: bool = None,
    tokenizer_regex: str = None,
    stopwords: Sequence[str] = None,
    text_flags: Dict = None,
    token_flags: Dict = None,
    memory: bool = False,
    verbose: bool = False,
    text_column=None,
    token_column=None,
):

    steps = list()
    if not text_column:
        text_columns = ("text",)
    else:
        text_columns = (text_column,)

    if not token_column:
        token_columns = ("tokens",)
    else:
        token_columns = (token_column,)

    # Normalizer
    if form:
        normalizer = Normalizer(
            form=form,
            lowercase=lowercase,
            input_columns=text_columns,
            output_columns=text_columns,
        )
        steps.append(("normalizer", normalizer))

    # Text Flagger
    if text_flags:
        text_flagger = DeterministicTextFlagger(
            text_flags=text_flags,
            input_columns=text_columns,
            output_columns=text_columns,
        )
        steps.append(("text_flagger", text_flagger))

    # Tokenizer
    if not tokenizer_regex:
        raise AttributeError("Tokenizer regexp is mandatory")
    else:
        tokenizer = RegexTokenizer(
            tokenizer_regex=tokenizer_regex,
            stopwords=stopwords,
            input_columns=text_columns,
            output_columns=token_columns,
        )
        steps.append(("tokenizer", tokenizer))

    # Token Flagger
    if token_flags:
        token_flagger = FlashtextTokenFlagger(
            token_flags=token_flags,
            input_columns=token_columns,
            output_columns=token_columns,
        )
        steps.append(("token_flagger", token_flagger))

    return MelusinePipeline(steps=steps, memory=memory, verbose=verbose)


def make_tokenizer_from_config(
    config, memory=False, verbose=False, text_column=None, token_column=None
):
    form = config["tokenizer"]["normalization"]
    lowercase = config["tokenizer"]["lowercase"]
    tokenizer_regex = config["tokenizer"]["tokenizer_regex"]
    if config["tokenizer"].get("remove_stopwords"):
        stopwords = config["tokenizer"]["stopwords"]
    else:
        stopwords = None
    token_flags = {config["tokenizer"]["name_flag"]: config["tokenizer"]["names"]}
    text_flags = config["tokenizer"]["flag_dict"]
    text_flags.update(config["tokenizer"]["collocations_dict"])

    return make_tokenizer(
        form=form,
        lowercase=lowercase,
        tokenizer_regex=tokenizer_regex,
        stopwords=stopwords,
        text_flags=text_flags,
        token_flags=token_flags,
        memory=memory,
        verbose=verbose,
        text_column=text_column,
        token_column=token_column,
    )
