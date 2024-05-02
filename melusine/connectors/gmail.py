import base64
import logging
import mimetypes
import os
from email import message, policy
from email.parser import BytesParser
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from tqdm import tqdm

logger = logging.getLogger(__name__)


SCOPES: List[str] = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.modify"]


class GmailConnector:
    """
    Connector to Gmail Mailboxs.
    This class contains methods suited for automated emails routing.
    """

    def __init__(
        self,
        token_json_path: Optional[str] = None,
        routing_label: Optional[str] = None,
        correction_label: Optional[str] = None,
        done_label: Optional[str] = None,
        target_column: str = "target",
    ):
        self.target_column = target_column

        # Connect to mailbox
        self.credentials: Credentials = self.get_credentials(token_json_path=token_json_path)
        self.service = build("gmail", "v1", credentials=self.credentials)
        self.labels: List[Dict[str, str]] = self._get_labels()

        # Setup correction folder and done folder
        self.routing_label: Optional[str] = self._check_or_create_label(routing_label)
        self.correction_label: Optional[str] = self._check_or_create_label(correction_label)
        self.done_label: Optional[str] = self._check_or_create_label(done_label)

        self.mailbox_address = self.service.users().getProfile(userId="me").execute()["emailAddress"]
        logger.info(f"Connected to mailbox: {self.mailbox_address}.")

    def __repr__(self) -> str:
        return (
            f"GmailConnector(routing_label={self.routing_label}, correction_label={self.correction_label},"
            + f" done_label={self.done_label}), connected to {self.mailbox_address}"
        )

    @staticmethod
    def get_credentials(token_json_path: Optional[str] = None):
        """TODO

        Args:
            token_json_path (Optional[str], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if token_json_path is not None and os.path.exists(token_json_path):
            creds: Credentials = Credentials.from_authorized_user_file("token.json", SCOPES)
            if creds.valid is False:
                creds.refresh(Request())
            return creds

        flow = InstalledAppFlow.from_client_secrets_file(
            # your creds file here. Please create json file as here https://cloud.google.com/docs/authentication/getting-started
            "credentials.json",
            SCOPES,
        )
        creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
        logger.info(f"gmail token.json saved at: {os.getcwd()}")

    def _get_labels(self) -> List[Dict]:
        """Retrieve all current labels in mailbox

        Returns:
            List[Dict]: List of labels dict
        """
        labels = self.service.users().labels().list(userId="me").execute()["labels"]
        return labels

    def _check_or_create_label(self, label_name: Optional[str] = None) -> Optional[str]:
        """Check label existance, if not, ask to create it.

        Args:
            label_name (Optional[str], optional): The label name to check. Defaults to None.

        Returns:
            Optional[str]: The label name
        """
        if label_name is None:
            return None

        all_labels_upper: List[str] = [label["name"].upper() for label in self.labels]
        if label_name.upper() in all_labels_upper:
            return label_name

        logger.warning(
            f"Label {label_name} does not exist in current labels list: {all_labels_upper}.\n"
            + "Would you like to create it? (Y/N)"
        )
        choice: str = input()
        if "Y" in choice.upper():
            result: Dict[str, str] = self.create_label(label_name)
            self.labels.append(result)
            return label_name
        raise ValueError(f"Label {label_name} does not exist.")

    def create_label(self, label_name: str) -> Dict[str, str]:
        """Create a label into connected mailbox

        Args:
            label_name (str): name of the new label

        Returns:
            Dict[str, str]: return from the api with label and its informations
        """
        label = self.service.users().labels().create(userId="me", body=dict(name=label_name)).execute()
        logger.info(f"Label {label_name} has been created.")
        return label

    def extract_from_parsed_mail(self, parsed_email: message.Message) -> Dict[str, Any]:
        """Extract body and attachments from the parsed email

        Args:
            parsed_email (message.Message): Message object containg all the email and its informations

        Returns:
            Dict[str, Any]: `body` key and `attachments_list` key with value inside the parsed email
        """
        body: str = ""
        if parsed_email.is_multipart():
            for part in parsed_email.walk():
                content_type = part.get_content_type()
                if "text/plain" in content_type:
                    bytes_string = part.get_payload(decode=True)
                    charset = part.get_content_charset("iso-8859-1")
                    body += bytes_string.decode(charset, "replace")
        else:
            bytes_string = parsed_email.get_payload(decode=True)
            charset = parsed_email.get_content_charset("iso-8859-1")
            body += bytes_string.decode(charset, "replace")

        attachments_list: List[Dict] = []
        for part in parsed_email.iter_attachments():  # type: ignore
            attachments_list.append(
                {
                    "filename": part.get_filename(),
                    "type": part.get_content_type(),
                    "data": part.get_payload(decode=True),
                }
            )
        return {"body": body, "attachments_list": attachments_list}

    def _extract_email_attributes(self, message_id: str) -> Dict:
        """Return formatted attributes for the considered email

        Args:
            message_id (str): id of the mail to consider

        Returns:
            Dict: formatted output of the email
        """
        msg_raw: Dict[str, Any] = (
            self.service.users().messages().get(id=message_id, userId="me", format="raw").execute()
        )
        parsed_email: message.Message = BytesParser(policy=policy.default).parsebytes(
            base64.urlsafe_b64decode(msg_raw["raw"])
        )

        infos: Dict[str, Any] = self.extract_from_parsed_mail(parsed_email)
        email_dict = {
            "message_id": message_id,
            "body": infos["body"],
            "header": parsed_email["Subject"],
            "date": parsed_email["Date"],
            "from": parsed_email["From"],
            "to": parsed_email["To"],
            "attachment": infos["attachments_list"],
        }
        return email_dict

    def get_emails(
        self,
        max_emails: int = 100,
        target_labels: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """TODO

        Args:
            max_emails (int, optional): _description_. Defaults to 100.
            target_labels (List[str], optional): _description_. Defaults to None.
            start_date (Optional[str], optional): _description_. Defaults to None.
            end_date (Optional[str], optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        logger.info("Reading new emails for mailbox")
        if target_labels is None:
            target_labels = ["INBOX"]
        target_label_id: List[str] = [item["id"] for item in self.labels if item["name"] in target_labels]
        q = ""
        if start_date is not None:
            q += f"after:{start_date} "

        if end_date is not None:
            q += f"before:{end_date}"

        all_new_data = (
            self.service.users()
            .messages()
            .list(userId="me", maxResults=max_emails, labelIds=target_label_id, q=q)
            .execute()
        )

        logger.info("Please wait while loading messages")
        new_emails: List[Dict] = [self._extract_email_attributes(x["id"]) for x in tqdm(all_new_data["messages"])]
        df_new_emails = pd.DataFrame(new_emails)

        logger.info(f"Read '{len(new_emails)}' new emails")
        return df_new_emails

    def _move_to(
        self,
        emails_id: List[str],
        label_to_move_on: Optional[str],
        attribute_class_to_set_error: str,
        func_name_error: str,
    ):
        """TODO

        Args:
            emails_id (List[str]): _description_
            label_to_move_on (Optional[str]): _description_
            attribute_class_to_set_error (str): _description_
            func_name_error (str): _description_

        Raises:
            AttributeError: _description_
        """
        if label_to_move_on is None:
            raise AttributeError(
                f"You need to set the class attribute `{attribute_class_to_set_error}` to use `{func_name_error}`."
            )
        label_id = next((label["id"] for label in self.labels if label["name"] == label_to_move_on), None)
        body = {"addLabelIds": [label_id]}
        for email_id in emails_id:
            self.service.users().messages().modify(id=email_id, userId="me", body=body).execute()
        logger.info(f"Moved {len(emails_id)} emails to {label_to_move_on} label.")

    def move_to_done(self, emails_id: List[str]) -> None:
        """Move emails to done label

        Args:
            emails_id (List[str]): List of emails id to move to done label
        """
        self._move_to(emails_id, self.done_label, "done_label", "move_to_done")

    def route_emails(
        self,
        classified_emails: pd.DataFrame,
        id_column: str = "message_id",
    ) -> None:
        """Function to route emails to mailbox folders.

        Args:
            classified_emails (pd.DataFrame): DataFrame containing emails message_id and target folder
            id_column (str, optional): Name of the DataFrame column containing message ids. Defaults to "message_id".
        """
        target_column = self.target_column
        target_labels = classified_emails[target_column].unique().tolist()

        for label in target_labels:
            mask = classified_emails[target_column] == label
            mids_to_move = classified_emails[mask][id_column]
            self._move_to(mids_to_move, label, label, "route_emails")
            logger.info(f"Moving {mids_to_move.size} emails to folder '{label}'")

    def send_email(self, to: Union[str, List[str]], header: str, body: str, attachments: dict) -> None:
        """This method sends an email from the login address (attribute login_address).

        Args:
            to (Union[str, List[str]]): Address or list of addresses of email recipients
            header (str): Email header
            body (str): Email body
            attachments (dict): Dict containing attachment names as key and attachment file contents as values.
            Currently, the code is tested for DataFrame attachments only
        """
        if isinstance(to, str):
            to = [to]

        m = message.EmailMessage()
        m.set_content(body)

        m["To"] = to
        m["Subject"] = header

        if attachments:
            for filename, value in attachments.items():
                type_subtype, _ = mimetypes.guess_type(filename)
                if type_subtype is not None:
                    maintype, subtype = type_subtype.split("/")
                    m.add_attachment(value, filename=filename, maintype=maintype, subtype=subtype)

        # encoded message
        encoded_message = base64.urlsafe_b64encode(m.as_bytes()).decode()
        create_message = {"raw": encoded_message}
        send_message = self.service.users().messages().send(userId="me", body=create_message).execute()
        logger.info(f"Email sent to {to}, Message Id: {send_message['id']}")
