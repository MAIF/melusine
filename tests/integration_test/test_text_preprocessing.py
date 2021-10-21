import os
from tempfile import TemporaryDirectory

import joblib
from sklearn.pipeline import Pipeline

from melusine import config
from melusine.data.data_loader import load_email_data
from melusine.nlp_tools.normalizer import Normalizer
from melusine.nlp_tools.phraser import Phraser
from melusine.nlp_tools.text_flagger import DeterministicTextFlagger
from melusine.nlp_tools.token_flagger import FlashtextTokenFlagger
from melusine.nlp_tools.tokenizer import RegexTokenizer


def test_text_preprocessing():
    input_df = load_email_data()
    input_col = "body"

    pipeline = Pipeline(
        steps=[
            ("normalizer", Normalizer(input_columns=input_col)),
            (
                "text_flagger",
                DeterministicTextFlagger(
                    input_columns=input_col,
                    text_flags=config["text_flagger"]["text_flags"],
                ),
            ),
            ("tokenizer", RegexTokenizer(input_columns=input_col)),
            (
                "token_flagger",
                FlashtextTokenFlagger(
                    token_flags=config["token_flagger"]["token_flags"],
                ),
            ),
            ("phraser", Phraser()),
        ]
    )

    pipeline.fit_transform(input_df)
    _ = pipeline.transform(input_df)

    with TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "pipe")
        joblib.dump(pipeline, filepath)
        pipeline_reload = joblib.load(filepath)

    _ = pipeline_reload.transform(input_df)
    assert True
