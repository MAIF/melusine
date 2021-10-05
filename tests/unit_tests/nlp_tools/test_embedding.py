import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory

from melusine.nlp_tools.embedding import Embedding


def test_embedding():
    df = pd.DataFrame(
        {
            "tokens": [
                ["bonjour", "a", "vous"],
                ["bonjour", "bonjour"],
                ["bonjour", "tout", "le", "monde"],
            ]
        }
    )

    w2v = Embedding(min_count=2, input_columns="tokens")
    w2v.fit(df)

    bonjour_embeddings = w2v.embeddings_["bonjour"]
    assert isinstance(bonjour_embeddings, np.ndarray)

    with TemporaryDirectory() as tmpdir:
        w2v.save(path=tmpdir)
        w2v_reload = Embedding.load(tmpdir)

    bonjour_embeddings = w2v_reload.embeddings_["bonjour"]
    assert isinstance(bonjour_embeddings, np.ndarray)
