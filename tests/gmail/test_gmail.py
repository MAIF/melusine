import logging
import os
import pytest

import pandas as pd
from unittest.mock import MagicMock, patch


HttpRequestMock = pytest.importorskip('googleapiclient.http.HttpRequestMock')
from google.oauth2.credentials import Credentials
from melusine.connectors.gmail import GmailConnector


def return_value(resp, content):
    return content


@pytest.fixture
def mocked_gc():
    with patch("melusine.connectors.gmail.build") as mock_build:
        with patch("melusine.connectors.gmail.Credentials.from_authorized_user_file") as mock_creds_from_file:
            with patch("melusine.connectors.gmail.os.path.exists") as mock_exists:
                mock_exists.return_value = True
                mock_service = MagicMock()
                mock_service.users().getProfile.return_value = HttpRequestMock(
                    None, {"emailAddress": "test@example.com"}, return_value
                )
                mock_service.users().labels().list.return_value = HttpRequestMock(
                    None,
                    {
                        "labels": [
                            {"id": "INBOX", "name": "INBOX", "type": "system"},
                            {
                                "id": "TRASH",
                                "name": "TRASH",
                                "messageListVisibility": "hide",
                                "labelListVisibility": "labelHide",
                                "type": "system",
                            },
                            {"id": "UNREAD", "name": "UNREAD", "type": "system"},
                        ]
                    },
                    return_value,
                )
                mock_build.return_value = mock_service
                mock_creds_from_file.return_value = Credentials("dummy")

                return GmailConnector(token_json_path="token.json", done_label="TRASH", target_column="target")


@pytest.fixture
def fake_image():
    image_data = b""
    width = height = 100
    for _ in range(height):
        row_data = b"\xff" * (width * 3)
        image_data += row_data

    return image_data


def return_value(resp, content):
    return content


@patch("melusine.connectors.gmail.build")
@patch("melusine.connectors.gmail.Credentials.from_authorized_user_file")
@patch("melusine.connectors.gmail.os.path.exists")
def test_init(mock_exists, mock_creds_from_file, mock_build, caplog):
    # Mocking necessary objects and methods
    mock_exists.return_value = True
    mock_service = MagicMock()
    mock_service.users().getProfile.return_value = HttpRequestMock(
        None, {"emailAddress": "test@example.com"}, return_value
    )
    mock_service.users().labels().list.return_value = HttpRequestMock(
        None,
        {
            "labels": [
                {"id": "INBOX", "name": "INBOX", "type": "system"},
                {
                    "id": "TRASH",
                    "name": "TRASH",
                    "messageListVisibility": "hide",
                    "labelListVisibility": "labelHide",
                    "type": "system",
                },
                {"id": "UNREAD", "name": "UNREAD", "type": "system"},
            ]
        },
        return_value,
    )
    mock_build.return_value = mock_service
    mock_creds_from_file.return_value = Credentials("dummy")

    # Creating an instance of GmailConnector
    with caplog.at_level(logging.DEBUG):
        gc = GmailConnector(token_json_path="token.json", done_label="TRASH", target_column="target")

    # Assertions
    assert len(gc.labels) == 3
    assert gc.done_label == "TRASH"
    assert gc.mailbox_address == "test@example.com"
    assert gc.target_column == "target"

    assert "Connected to mailbox:" in caplog.text

    assert str(gc) == "GmailConnector(done_label=TRASH, target_column=target), connected to test@example.com"


@patch("melusine.connectors.gmail.build")
@patch("melusine.connectors.gmail.InstalledAppFlow.from_client_secrets_file")
def test_init_without_creds(mock_flow, mock_build, caplog):
    # Mocking necessary objects and methods
    mock_service = MagicMock()
    mock_service.users().getProfile.return_value = HttpRequestMock(
        None, {"emailAddress": "test@example.com"}, return_value
    )
    mock_service.users().labels().list.return_value = HttpRequestMock(
        None,
        {
            "labels": [
                {"id": "INBOX", "name": "INBOX", "type": "system"},
                {
                    "id": "TRASH",
                    "name": "TRASH",
                    "messageListVisibility": "hide",
                    "labelListVisibility": "labelHide",
                    "type": "system",
                },
                {"id": "UNREAD", "name": "UNREAD", "type": "system"},
            ]
        },
        return_value,
    )
    mock_build.return_value = mock_service
    mock_flow.return_value.run_local_server.return_value = Credentials("dummy")

    # Creating an instance of GmailConnector
    with caplog.at_level(logging.DEBUG):
        gc = GmailConnector()

    # Assertions
    assert len(gc.labels) == 3
    assert gc.done_label is None
    assert gc.mailbox_address == "test@example.com"
    assert gc.target_column == "target"
    assert os.path.exists("token.json")
    os.remove("token.json")
    assert "gmail token.json saved at:" in caplog.text
    assert "Connected to mailbox:" in caplog.text

    assert str(gc) == "GmailConnector(done_label=None, target_column=target), connected to test@example.com"


def test_gc_get_emails(mocked_gc, simple_email_raw, caplog):
    mocked_gc.service.users().messages().list.return_value = HttpRequestMock(
        None, {"messages": [{"id": "123"}]}, return_value
    )
    mocked_gc.service.users().messages().get.return_value = HttpRequestMock(
        None,
        {
            "id": "123",
            "labelIds": ["INBOX"],
            "snippet": "Did it worked?",
            "sizeEstimate": 45200,
            "raw": simple_email_raw,
        },
        return_value,
    )
    with caplog.at_level(logging.DEBUG):
        df = mocked_gc.get_emails(1, None, "2024/01/01", "2024/05/03")

    assert "Please wait while loading messages" in caplog.text
    assert "Read '1' new emails" in caplog.text

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0].to_dict() == {
        "message_id": "123",
        "body": "This body should appeared!\n",
        "header": "Did it worked?",
        "date": "Thu, 02 May 2024 08:53:35 -0700",
        "from": "sender@example.com",
        "to": "recipient@example.com",
        "attachment": [],
    }


def test_gc_get_emails_complex_mail(mocked_gc, complex_email_raw, caplog):
    mocked_gc.service.users().messages().list.return_value = HttpRequestMock(
        None, {"messages": [{"id": "123"}]}, return_value
    )
    mocked_gc.service.users().messages().get.return_value = HttpRequestMock(
        None,
        {
            "id": "123",
            "labelIds": ["INBOX"],
            "snippet": "Did it worked?",
            "sizeEstimate": 45200,
            "raw": complex_email_raw,
        },
        return_value,
    )
    with caplog.at_level(logging.DEBUG):
        df = mocked_gc.get_emails(1, None, "2024/01/01", "2024/05/03")

    assert "Please wait while loading messages" in caplog.text
    assert "Read '1' new emails" in caplog.text

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0].to_dict() == {
        "message_id": "123",
        "body": "This is the body of the email.",
        "header": "Fake multipart email",
        "date": "Thu, 02 May 2024 08:53:35 -0700",
        "from": "sender@example.com",
        "to": "recipient@example.com",
        "attachment": [{"filename": "attachment.txt", "type": "application/octet-stream", "data": b"dummy text"}],
    }


def test_gc_get_emails_none(mocked_gc, simple_email_raw, caplog):
    mocked_gc.service.users().messages().list.return_value = HttpRequestMock(None, {}, return_value)
    with caplog.at_level(logging.DEBUG):
        df = mocked_gc.get_emails(1, None, "2024/01/01", "2024/05/03")

    assert "Please wait while loading messages" not in caplog.text
    assert "No emails with filters: target_labels=" in caplog.text

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert all([col in df.columns for col in ["message_id", "body", "header", "date", "from", "to", "attachment"]])


@patch("builtins.input", side_effect=["y", "n"])
def test_gc_check_or_create_label(mock_input, mocked_gc, caplog):
    mocked_gc.service.users().labels().create.return_value = HttpRequestMock(
        None,
        {
            "id": "Label_3",
            "name": "test",
            "messageListVisibility": "show",
            "labelListVisibility": "labelShow",
            "type": "user",
        },
        return_value,
    )
    assert len([label for label in mocked_gc.labels if label["name"] == "test"]) == 0
    with caplog.at_level(logging.DEBUG):
        label = mocked_gc._check_or_create_label("test")
    assert "does not exist in current labels list" in caplog.text

    assert label == "test"
    assert len([label for label in mocked_gc.labels if label["name"] == "test"]) == 1
    assert "Label test has been created" in caplog.text

    # With input to "n"
    with pytest.raises(ValueError, match="Label test2 does not exist."):
        mocked_gc._check_or_create_label("test2")


def test_gc_move_to_done(mocked_gc, caplog):
    mocked_gc.service.users().messages().modify.return_value = HttpRequestMock(None, {}, return_value)
    with caplog.at_level(logging.DEBUG):
        mocked_gc.move_to_done(["dummy_id"])

    assert "Moved 1 emails to 'TRASH' label." in caplog.text

    mocked_gc.done_label = None
    with pytest.raises(AttributeError, match="You need to set the class attribute `done_label` to use `move_to_done`."):
        mocked_gc.move_to_done(["dummy_id"])


def test_gc_move_to_error(mocked_gc, caplog):
    with pytest.raises(
        ValueError,
        match="Label 'not_existing_label' does not exist in self.labels. Make sure to specified a right label name.",
    ):
        mocked_gc.move_to(["dummy_id"], "not_existing_label")


def test_gc_route_emails(mocked_gc, caplog):
    mocked_gc.service.users().messages().modify.return_value = HttpRequestMock(None, {}, return_value)

    df = pd.DataFrame(
        {
            "message_id": ["123", "456"],
            "body": ["Body1", "Body2"],
            "header": ["Header1", "Header2"],
            "date": ["Thu, 02 May 2024 08:53:35 -0700", "Thu, 02 May 2024 10:00:00 -0700"],
            "from": ["sender2@example.com", "sender2@example.com"],
            "to": ["recipient@example.com", "recipient@example.com"],
            "attachment": [[], []],
            "target": ["TRASH", "UNREAD"],
        }
    )
    with caplog.at_level(logging.DEBUG):
        mocked_gc.route_emails(df)

    assert "Moved 1 emails to 'TRASH' label" in caplog.text
    assert "Moved 1 emails to 'UNREAD' label" in caplog.text


def test_gc_send_email(mocked_gc, fake_image, caplog):
    mocked_gc.service.users().messages().send.return_value = HttpRequestMock(None, {"id": "12456"}, return_value)

    with caplog.at_level(logging.DEBUG):
        mocked_gc.send_email(
            "melusine_testing.yopmail.com",
            "Testing Header",
            "Testing Body",
            {"attachment.jpg": fake_image},
        )

    assert "Email sent to melusine_testing@yopmail.com, Message Id: 12456"  in caplog.text
