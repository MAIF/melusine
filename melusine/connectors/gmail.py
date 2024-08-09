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
from googleapiclient.errors import HttpError
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GmailConnector:
    """
    Connector to Gmail Mailboxs.
    This class contains methods suited for automated emails routing.
    A credentials.json file is needed to sign in to google. To do so, follow these steps :
    https://medium.com/@preetipriyanka24/how-to-read-emails-from-gmail-using-gmail-api-in-python-20f7d9d09ae9
    Please do not forget to add the mail address to the list of allowed tester if the credentials stays in testing

    First use can only be:
    ```python
    # Assuming credentials.json is at root level
    gc = GmailConnector()
    # A pop up window will ask you to connect then the next connection will be:
    gc = GmailConnector("token.json")
    ```
    """

    SCOPES: List[str] = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.modify",
    ]

    def __init__(
        self,
        token_json_path: Optional[str] = None,
        credentials_json_path: str = "credentials.json",
        done_label: Optional[str] = None,
        target_column: str = "target",
    ):
        """
        Args:
            token_json_path (Optional[str], optional): `token.json` file path created after the first connection using
            `credentials.json`. If None, looking for credentials_json_path and sign in. Defaults to None.
            credentials_json_path (str, optional): file path for credentials.json delivered by google.
            Defaults to credentials.json at root.
            done_label (Optional[str], optional): Label name for the done situation. Defaults to None.
            target_column (str, optional): Name of the DataFrame column containing target label. Defaults to "target".
        """
        self.target_column: str = target_column

        # Connect to mailbox
        self.credentials: Credentials = self.get_credentials(
            token_json_path=token_json_path, credentials_json_path=credentials_json_path
        )
        self.service: Any = build("gmail", "v1", credentials=self.credentials)

        # Get and setup labels
        self.labels: List[Dict[str, str]] = self._get_labels()
        self.done_label: Optional[str] = self._check_or_create_label(done_label)

        self.mailbox_address: str = self.service.users().getProfile(userId="me").execute()["emailAddress"]
        logger.info(f"Connected to mailbox: {self.mailbox_address}.")

    def __repr__(self) -> str:
        """
        Returns:
            str: Reprensentation of the object
        """
        return (
            f"GmailConnector(done_label={self.done_label}, target_column={self.target_column}), "
            + f"connected to {self.mailbox_address}"
        )

    def get_credentials(self, credentials_json_path: str, token_json_path: Optional[str] = None) -> Credentials:
        """Retrieve credentials object to connect to Gmail using the `credentials.json` and generating the `token.json`
        if needed at root path.
        Please create json file as here https://cloud.google.com/docs/authentication/getting-started

        Args:
            credentials_json_path (str): Credentials file path delivered by Google to authenticate.
            token_json_path (Optional[str], optional): `token.json` file path created after the first connection using
            `credentials.json`. Defaults to None.

        Returns:
            Credentials: Credentials to connect to Gmail
        """
        # Get the token from the path
        if token_json_path is not None and os.path.exists(token_json_path):
            creds: Credentials = Credentials.from_authorized_user_file(token_json_path, self.SCOPES)
            if creds.valid is False:
                creds.refresh(Request())
            return creds

        # Ask for token to Google
        flow: InstalledAppFlow = InstalledAppFlow.from_client_secrets_file(
            credentials_json_path,
            self.SCOPES,
        )
        creds = flow.run_local_server(port=0)

        # Save the token for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

        logger.info(f"gmail token.json saved at: {os.getcwd()}")
        return creds

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
        try:
            label = self.service.users().labels().create(userId="me", body=dict(name=label_name)).execute()
            logger.info(f"Label {label_name} has been created.")
        except HttpError as error:
            if error.resp.status == 409:  # Conflict error if label already exists
                logger.error(f"Label '{label_name}' already exists.")
                return {}
            else:
                raise
        return label

    @staticmethod
    def extract_from_parsed_mail(parsed_email: message.Message) -> Dict[str, Any]:
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
        """Return formatted attributes for the considered email such as:
        - `message_id` field
        - `body` field
        - `header` field
        - `date` field
        - `from` field
        - `to` field
        - `attachment` field

        Args:
            message_id (str): id of the mail to consider

        Returns:
            Dict: formatted output of the email
        """

        # Get the raw message and create a Message object
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
        """Loads emails in descending date order based on target_labels. To see all available labels, use `self.labels`.
        If two label names are defined, retrieve all emails with both labels, e.g. ["TRASH", "INBOX"] will retrieve none
        These labels cannot be present simultaneously.

        For example, to get first 10 mails received in inbox and unreaded:
        ```python
        gc = GmailConnector("token.json", done_label="test")
        df = gc.get_emails(10, target_labels=["INBOX", "UNREAD"])
        df
        ```

        Args:
            max_emails (int, optional): Maximum number of emails to load. Defaults to 100.
            target_labels (List[str], optional): Label names list to fetch. Defaults to None. If None, fetch INBOX.
            start_date (Optional[str], optional): Filter date start, format YYYY/MM/DD. Defaults to None.
            end_date (Optional[str], optional): Filter date end, format YYYY/MM/DD. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing emails
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

        all_new_data: Dict[str, Any] = (
            self.service.users()
            .messages()
            .list(userId="me", maxResults=max_emails, labelIds=target_label_id, q=q)
            .execute()
        )

        if "messages" not in all_new_data:
            logger.info(
                f"No emails with filters: target_labels={target_labels}, start_date={start_date}, end_date={end_date}"
            )
            return pd.DataFrame(columns=["message_id", "body", "header", "date", "from", "to", "attachment"])
        logger.info("Please wait while loading messages")
        new_emails: List[Dict] = [self._extract_email_attributes(x["id"]) for x in tqdm(all_new_data["messages"])]
        df_new_emails = pd.DataFrame(new_emails)

        logger.info(f"Read '{len(new_emails)}' new emails")
        return df_new_emails

    def move_to(self, emails_id: List[str], label_to_move_on: str) -> None:
        """Generic method to move emails to a specified label

        Args:
            emails_id (List[str]): List of emails id to set the label
            label_to_move_on (str): Label name to set

        """
        label_id: Optional[str] = next(
            (label["id"] for label in self.labels if label["name"] == label_to_move_on), None
        )
        if label_id is None:
            raise ValueError(
                f"Label '{label_to_move_on}' does not exist in self.labels. Make sure to specified a right label name."
            )
        for email_id in emails_id:
            self.service.users().messages().modify(id=email_id, userId="me", body={"addLabelIds": [label_id]}).execute()

        logger.info(f"Moved {len(emails_id)} emails to '{label_to_move_on}' label.")

    def move_to_done(self, emails_id: List[str]) -> None:
        """Move emails to done label

        Args:
            emails_id (List[str]): List of emails id to move to done label
        """
        if self.done_label is None:
            raise AttributeError("You need to set the class attribute `done_label` to use `move_to_done`.")
        self.move_to(emails_id, self.done_label)

    def route_emails(self, classified_emails: pd.DataFrame, id_column: str = "message_id") -> None:
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
            self.move_to(mids_to_move, label)

    def send_email(self, to: Union[str, List[str]], header: str, body: str, attachments: Optional[Dict] = None) -> None:
        """This method sends an email from the login address (attribute login_address).

        Args:
            to (Union[str, List[str]]): Address or list of addresses of email recipients
            header (str): Email header
            body (str): Email body
            attachments (Optional[Dict], optional): Dict containing attachment names as key and attachment
            file contents as values. Defaults to None.
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
