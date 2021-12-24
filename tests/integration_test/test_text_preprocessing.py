import os
from tempfile import TemporaryDirectory

import joblib
from sklearn.pipeline import Pipeline

from melusine.data.data_loader import load_email_data
from melusine.nlp_tools.phraser import Phraser
from melusine.nlp_tools.tokenizer import Tokenizer


def test_text_preprocessing():
    input_df = load_email_data()
    input_col = "body"

    pipeline = Pipeline(
        steps=[
            ("tokenizer", Tokenizer(input_column=input_col)),
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
