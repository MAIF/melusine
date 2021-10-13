import logging

from melusine.nlp_tools.normalizer import Normalizer
from melusine.core.pipeline import MelusinePipeline
from melusine.nlp_tools.text_flagger import (
    DeterministicTextFlagger,
)
from melusine.nlp_tools.token_flagger import FlashtextTokenFlagger
from melusine.nlp_tools.tokenizer import RegexTokenizer

logger = logging.getLogger(__name__)


def make_tokenizer_from_config(
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
    normalizer = Normalizer.from_config(
        config_key="normalizer",
        input_columns=text_columns,
        output_columns=text_columns,
    )
    steps.append(("normalizer", normalizer))

    # Text Flagger
    text_flagger = DeterministicTextFlagger.from_config(
        config_key="text_flagger",
        input_columns=text_columns,
        output_columns=text_columns,
    )
    steps.append(("text_flagger", text_flagger))

    # Tokenizer
    tokenizer = RegexTokenizer.from_config(
        config_key="tokenizer",
        input_columns=text_columns,
        output_columns=token_columns,
    )
    steps.append(("tokenizer", tokenizer))

    # Token Flagger
    token_flagger = FlashtextTokenFlagger.from_config(
        config_key="token_flagger",
        input_columns=token_columns,
        output_columns=token_columns,
    )
    steps.append(("token_flagger", token_flagger))

    return MelusinePipeline(steps=steps, memory=memory, verbose=verbose)
