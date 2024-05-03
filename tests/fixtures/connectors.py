from unittest.mock import MagicMock, patch

import pytest
from google.oauth2.credentials import Credentials
from googleapiclient.http import HttpRequestMock

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
