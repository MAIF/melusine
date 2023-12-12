import pytest


@pytest.fixture
def conf_normalizer():
    return {
        "form": "NFKD",
        "input_columns": ["text"],
        "lowercase": True,
        "output_columns": ["text"],
    }


@pytest.fixture
def conf_tokenizer():
    return {
        "stopwords": ["le", "les"],
        "tokenizer_regex": '\\w+(?:[\\?\\-\\"_]\\w+)*',
    }


@pytest.fixture
def conf_phraser():
    return {
        "input_columns": ["tokens"],
        "output_columns": ["tokens"],
        "threshold": 10,
        "min_count": 10,
    }
