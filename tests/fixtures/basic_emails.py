import base64
from email.message import EmailMessage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.policy import default

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


@pytest.fixture
def simple_email_raw():
    email_message = EmailMessage()
    email_message["From"] = "sender@example.com"
    email_message["To"] = "recipient@example.com"
    email_message["Subject"] = "Did it worked?"
    email_message["Date"] = "Thu, 02 May 2024 08:53:35 -0700"
    email_message.set_content("This body should appeared!")

    email_bytes = email_message.as_bytes(policy=default)

    return base64.urlsafe_b64encode(email_bytes)


@pytest.fixture
def complex_email_raw():
    msg = MIMEMultipart()
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Subject"] = "Fake multipart email"
    msg["Date"] = "Thu, 02 May 2024 08:53:35 -0700"

    text_part = MIMEText("This is the body of the email.")
    msg.attach(text_part)

    # Add attachment
    attachment_part = MIMEApplication("dummy text", Name="attachment.txt")
    attachment_part["Content-Disposition"] = 'attachment; filename="attachment.txt"'
    msg.attach(attachment_part)

    email_bytes = msg.as_bytes(policy=default)

    return base64.urlsafe_b64encode(email_bytes).decode("utf-8")
