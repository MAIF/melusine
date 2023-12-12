"""
Setup tests and import fixtures
"""
import numpy as np
import pytest

from melusine import config

# Declare fixtures
pytest_plugins = [
    "tests.fixtures.backend",
    "tests.fixtures.basic_emails",
    "tests.fixtures.docs",
    "tests.fixtures.pipelines",
    "tests.fixtures.processors",
]


# =============== Generic fixtures ===============
# Print statements inside fixtures are only visible when running
# pytest -s


@pytest.fixture(scope="session")
def df_emails():
    from melusine.data import load_email_data
    from melusine.processors import RegexTokenizer

    # Load data
    df_emails = load_email_data(type="full")

    # Tokenize text
    tokenizer = RegexTokenizer(input_columns="body")
    df_emails = tokenizer.transform(df_emails)

    # Add mock meta features
    df_emails["test_meta__A"] = np.random.randint(0, 2, size=len(df_emails))
    df_emails["test_meta__B"] = np.random.randint(0, 2, size=len(df_emails))

    return df_emails


# =============== Fixtures with "function" scope ===============
@pytest.fixture(scope="function")
def reset_melusine_config():
    """
    When a test modifies the melusine configuration, this fixture can be used to reset the config.
    """
    # Code executed before the test starts
    pass

    # Run the test
    yield

    # Code executed after the test ends
    config.reset()


@pytest.fixture(scope="function")
def use_test_config(conf_normalizer, conf_tokenizer, conf_phraser):
    """
    Add test configurations.
    """
    # Code executed before the test starts
    test_conf_dict = config.dict()

    test_conf_dict["test_tokenizer"] = conf_tokenizer
    test_conf_dict["test_normalizer"] = conf_normalizer
    test_conf_dict["test_phraser"] = conf_phraser

    config.reset(config_dict=test_conf_dict)

    # Run the test
    yield

    # Code executed after the test ends
    config.reset()


@pytest.fixture(scope="function")
def use_dict_backend():
    """
    When a test modifies the melusine backend, this fixture can be used to reset the backend.
    """
    from melusine.backend.active_backend import backend

    # =========== Code executed before the test starts ===========
    # Use a dict backend to test a pipeline message by message
    backend.reset("dict")

    # =========== Run the test ===========
    yield

    # =========== Code executed after the test ends ===========
    backend.reset()
