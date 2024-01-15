import pandas as pd
import pytest


@pytest.fixture
def text_basic_hello_world():
    return {"text": "Hello world", "header": "Hello"}


@pytest.fixture
def text_basic_with_accent():
    return {
        "text": "Bonjour\nPouvez-vous résilier mon contrat?\nJean Dupont",
        "header": "Demande de résiliation",
    }


@pytest.fixture
def email_basic_hello_world():
    return {
        "body": "Hello world",
        "header": "Hello",
        "chanel": "mail",
        "from": "jo@gmail.com",
        "to": ["test@maif.fr"],
        "bal": "",
        "date": "",
        "ged_chanel": "",
        "pli_id": "",
        "flux_id": "",
        "entity": "",
        "soc_type": "",
        "soc_num": "",
        "nb_documents": 0,
    }


@pytest.fixture
def email_basic_with_accent():
    return {
        "body": "Bonjour\nPouvez-vous résilier mon contrat?\nJean Dupont",
        "header": "Demande de résiliation",
        "chanel": "mail",
        "from": "jo@gmail.com",
        "to": ["test@maif.fr"],
        "bal": "",
        "date": "",
        "ged_chanel": "",
        "pli_id": "",
        "flux_id": "",
        "entity": "",
        "soc_type": "",
        "soc_num": "",
        "nb_documents": 0,
    }


@pytest.fixture
def dataframe_basic(text_basic_hello_world, text_basic_with_accent):
    return pd.DataFrame(
        [
            text_basic_hello_world,
            text_basic_with_accent,
        ]
    )


@pytest.fixture
def email_dataframe_basic(email_basic_hello_world, email_basic_with_accent):
    return pd.DataFrame(
        [
            email_basic_hello_world,
            email_basic_with_accent,
        ]
    )
