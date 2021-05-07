import logging
import re

import pandas as pd
from typing import List, Union

logger = logging.getLogger(__name__)
logging.getLogger("exchangelib").setLevel(logging.WARNING)

try:
    from exchangelib import (
        Credentials,
        Account,
        Folder,
        Message,
        FileAttachment,
        HTMLBody,
        Configuration,
        FaultTolerance,
    )  # noqa
    from exchangelib.errors import ErrorFolderNotFound  # noqa
except ModuleNotFoundError:
    logger.exception(
        "To use the Melusine ExchangeConnector, you need to install the exchangelib package"
        "pip install exchangelib"
    )
    raise


class ExchangeConnector:
    """
    Connector to Outlook Exchange Mailboxs.
    This class contains methods suited for automated emails routing.
    """

    def __init__(
        self,
        mailbox_address: str,
        password: str,
        max_wait: int = 60,
        routing_folder_path: str = None,
        correction_folder_path: str = None,
        done_folder_path: str = None,
        user_address=None,
        target_column="target",
    ):
        """
        Parameters
        ----------
        mailbox_address: str
            Mailbox you need to connect to.
        password: str
            Password to login to the Exchange mailbox
        max_wait: int
            Maximum time (in s) to wait when connecting to mailbox
        routing_folder_path: str
            Path of the base routing folder
        correction_folder_path: str
            Path of the base correction folder
        done_folder_path: str
            Path of the Done folder
        target_column: str
            Name of the DataFrame column containing target folder names
        """
        self.mailbox_address = mailbox_address
        self.folder_list = None
        self.target_column = target_column
        self.user_address = user_address or mailbox_address

        # Connect to mailbox
        self.credentials = Credentials(self.user_address, password)
        self.exchangelib_config = Configuration(
            retry_policy=FaultTolerance(max_wait=max_wait), credentials=self.credentials
        )
        self.mailbox_account = Account(
            self.mailbox_address,
            credentials=self.credentials,
            autodiscover=True,
            config=self.exchangelib_config,
        )

        # Setup correction folder and done folder
        self.routing_folder_path = routing_folder_path
        self.correction_folder_path = correction_folder_path
        self.done_folder_path = done_folder_path

        logger.info(
            f"Connected to mailbox {self.mailbox_address} as user {self.user_address}"
        )

    def _get_mailbox_path(self, path):
        # Default to inbox
        if not path:
            return self.mailbox_account.inbox

        # Start mailbox path from root folder
        if re.match("/?root", path, flags=re.I):
            path = re.split("/?root/?", path, flags=re.I)[1]
            mailbox_path = self.mailbox_account.root

        # Start mailbox path from inbox folder
        else:
            mailbox_path = self.mailbox_account.inbox

        # Build mailbox path
        folders = path.split("/")
        for folder in folders:
            if folder == "..":
                mailbox_path = mailbox_path.parent
            else:
                mailbox_path = mailbox_path / folder

        return mailbox_path

    @property
    def routing_folder_path(self):
        return self._routing_folder_path

    @routing_folder_path.setter
    def routing_folder_path(self, routing_folder_path: str):
        if not routing_folder_path:
            self._routing_folder_path = None
            self.routing_folder = None
        else:
            self._routing_folder_path = routing_folder_path
            self.routing_folder = self._get_mailbox_path(routing_folder_path)

    @property
    def done_folder_path(self):
        return self._done_folder_path

    @done_folder_path.setter
    def done_folder_path(self, done_folder_path: str):
        if not done_folder_path:
            self._done_folder_path = None
            self.done_folder = None
        else:
            self._done_folder_path = done_folder_path
            self.done_folder = self._get_mailbox_path(done_folder_path)

    @property
    def correction_folder_path(self):
        return self._correction_folder_path

    @correction_folder_path.setter
    def correction_folder_path(self, correction_folder_path: str):
        if not correction_folder_path:
            self._correction_folder_path = None
            self.correction_folder = None
        else:
            self._correction_folder_path = correction_folder_path
            self.correction_folder = self._get_mailbox_path(correction_folder_path)

    @property
    def sender_account(self):
        if not self.user_address:
            raise AttributeError(
                f"You need to set the attribute user_address to send emails"
            )

        return Account(
            self.user_address,
            credentials=self.credentials,
            autodiscover=True,
            config=self.exchangelib_config,
        )

    def create_folders(self, folder_list: List[str], base_folder_path: str = None):
        """Create folders in the mailbox.

        Parameters
        ----------
        folder_list : list
            Create folders in the mailbox
        base_folder_path : str
            New folders will be created inside at path base_folder_path (Defaults to inbox)
        """
        self.folder_list = folder_list

        # Setup base folder
        base_folder = self._get_mailbox_path(base_folder_path)

        # Check existing folders
        existing_folders = [f.name for f in base_folder.children]

        # Create new folders
        for folder_name in folder_list:
            if folder_name not in existing_folders:
                f = Folder(parent=base_folder, name=folder_name)
                f.save()
                logger.info(
                    f"Created subfolder {folder_name} in folder {base_folder_path}"
                )

    def get_emails(
        self, max_emails: int = 100, base_folder_path: str = None
    ) -> pd.DataFrame:
        """
        Load emails in the inbox.

        Parameters
        ----------
        max_emails: int
             Maximum number of emails to load
        base_folder_path: str
            Path to folder to fetch

        Returns
        -------
        df_new_emails: pandas.DataFrame
            DataFrame containing nex emails
        """
        logger.info(f"Reading new emails for mailbox '{self.mailbox_address}'")
        base_folder = self._get_mailbox_path(base_folder_path)

        all_new_data = (
            base_folder.all()
            .only(
                "message_id",
                "datetime_sent",
                "sender",
                "to_recipients",
                "subject",
                "text_body",
                "attachments",
            )
            .order_by("datetime_received")[:max_emails]
        )

        new_emails = [
            self._extract_email_attributes(x)
            for x in all_new_data
            if isinstance(x, Message)
        ]
        df_new_emails = pd.DataFrame(new_emails)

        logger.info(f"Read '{len(new_emails)}' new emails")
        return df_new_emails

    @staticmethod
    def _extract_email_attributes(email_item: Message) -> dict:
        """
        Load email attributes of interest such as:
        - `message_id` field
        - `body` field
        - `header` field
        - `date` field
        - `from` field
        - `to` field
        - `attachment` field

        Parameters
        ----------
        email_item: exchangelib.Message
            Exchange Message object

        Returns
        -------
        email_dict: dict
            Dict with email attributes of interest
        """
        if not email_item.to_recipients:
            to_list = list()
        else:
            to_list = [i.email_address for i in email_item.to_recipients]

        if not email_item.attachments:
            attachments_list = None
        else:
            attachments_list = [i.name for i in email_item.attachments]

        email_dict = {
            "message_id": email_item.message_id,
            "body": email_item.text_body or "",
            "header": email_item.subject or "",
            "date": email_item.datetime_sent.isoformat(),
            "from": email_item.sender.email_address or None,
            "to": to_list,
            "attachment": attachments_list,
        }
        return email_dict

    def route_emails(
        self,
        classified_emails: pd.DataFrame,
        raise_missing_folder_error: bool = False,
        id_column: str = "message_id",
    ):
        """
        Function to route emails to mailbox folders.

        Parameters
        ----------
        classified_emails: pandas.DataFrame
            DataFrame containing emails message_id and target folder
        raise_missing_folder_error: bool
            Whether an error should be raised when a target folder is missing
        id_column: str
            Name of the DataFrame column containing message ids
        """
        target_column = self.target_column
        target_folders = classified_emails[target_column].unique().tolist()
        base_folder = self.routing_folder

        for folder in target_folders:
            try:
                destination_folder = base_folder / folder
            except ErrorFolderNotFound:
                if raise_missing_folder_error:
                    logger.exception(f"Mailbox (sub)folder '{folder}' not found")
                    raise
                else:
                    logger.warning(f"Mailbox (sub)folder '{folder}' not found")
                    continue

            mask = classified_emails[target_column] == folder
            mids_to_move = classified_emails[mask][id_column]
            items = self.mailbox_account.inbox.filter(message_id__in=mids_to_move).only(
                "id", "changekey"
            )
            self.mailbox_account.bulk_move(
                ids=items, to_folder=destination_folder, chunk_size=5
            )
            logger.info(f"Moving {mids_to_move.size} emails to folder '{folder}'")

    def get_corrections(
        self,
        max_emails: int = 100,
        ignore_list: List[str] = tuple(),
    ) -> pd.DataFrame:
        """
        When mailbox users find misclassified emails, they should move them to correction folders.
        This method collects the emails placed in the correction folders.

        Parameters
        ----------
        max_emails: int
             Maximum number of emails to fetch at once
        ignore_list: list
             List of folders that should be ignored when fetching emails

        Returns
        -------
        df_corrected_emails: pandas.DataFrame
            DataFrame containing the misclassified emails ids and associated correction folder
        """
        if not self.correction_folder_path:
            raise AttributeError(
                "You need to specify a correction_folder_name when instantiating the ExchangeConnector class."
            )

        logger.info(
            f"Reading corrected emails from folder and {self.correction_folder}"
        )

        # Get correction folders
        categories = [
            e.name for e in self.correction_folder.children if e.name not in ignore_list
        ]

        # Load corrected emails
        all_corrected_emails = list()
        for folder_name in categories:
            folder = self.correction_folder / folder_name
            messages = (
                folder.all()
                .only(
                    "message_id",
                    "datetime_sent",
                    "sender",
                    "to_recipients",
                    "subject",
                    "text_body",
                    "attachments",
                )
                .order_by("datetime_received")[:max_emails]
            )
            emails = [
                self._extract_email_attributes(m)
                for m in messages
                if isinstance(m, Message)
            ]

            # Add correction folder to email attributes
            for item in emails:
                item.update({"correction": folder_name})

            all_corrected_emails.extend(emails)
            logger.info(f"Found {len(emails)} corrected emails in folder {folder}")

        logger.info(f"Found {len(all_corrected_emails)} corrected emails in total")
        df_corrected_emails = pd.DataFrame(all_corrected_emails)

        return df_corrected_emails

    def move_to_done(self, emails_id: List[str]):
        """
        Once the corrected emails have been processed, they can be moved to a "Done" folder.

        Parameters
        ----------
        emails_id: list
            List of emails IDs to be moved to the done folder.
        """
        if (not self.done_folder_path) or (not self.correction_folder_path):
            raise AttributeError(
                "You need to specify a done_folder_name and a correction_folder_name"
                "when instantiating the ExchangeConnector class."
            )
        # Collect corrected emails
        items = self.correction_folder.children.filter(message_id__in=emails_id).only(
            "id", "changekey"
        )
        n_items = items.count()

        # Move to done folder
        self.mailbox_account.bulk_move(
            ids=items, to_folder=self.done_folder, chunk_size=5
        )
        logger.info(
            f"Moved {n_items} corrected emails to the folder {self.done_folder_path}"
        )

    def list_subfolders(self, base_folder_path: str = None):
        """
        List the sub-folders of the specified folder.

        Parameters
        ----------
        base_folder_path: str
            Path to folder to be inspected
        """
        base_folder = self._get_mailbox_path(base_folder_path)
        return [f.name for f in base_folder.children]

    def send_email(
        self, to: Union[str, List[str]], header: str, body: str, attachments: dict
    ):
        """
        This method sends an email using the class attribute "sender_address" as sender.

        Parameters
        ----------
        to: str or list
            Address or list of addresses of email recipients
        header: str
            Email header
        body: str
             Email body
        attachments: dict
            Dict containing attachment names as key and attachment file contents as values.
            Currently, the code is tested for DataFrame attachments only.
        """
        if isinstance(to, str):
            to = [to]

        # Prepare Message object
        m = Message(
            account=self.sender_account,
            subject=header,
            body=HTMLBody(body),
            to_recipients=to,
        )
        if attachments:
            for key, value in attachments.items():
                m.attach(FileAttachment(name=key, content=bytes(value, "utf-8")))

        # Send email
        m.send()
        logger.info(f"Email sent from address '{self.user_address}'")
