import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.getLogger('exchangelib').setLevel(logging.WARNING)

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
    logger.error(
        "To use the Melusine ExchangeConnector, you need to install the exchangelib package"
    )


class ExchangeConnector:
    """
    Connector to Outlook Exchange Mailbox.
    This class contains methods suited for automated emails routing.
    """

    def __init__(
        self,
        mailbox_address,
        password,
        max_wait=60,
        correction_folder_name=None,
        done_folder_name=None,
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
        correction_folder_name: str
            Name of the base correction folder
        done_folder_name: str
            Name of the Done folder
        """
        self.mailbox_address = mailbox_address
        self.folder_list = None

        # Connect to mailbox
        credentials = Credentials(self.mailbox_address, password)
        self.exchangelib_config = Configuration(
            retry_policy=FaultTolerance(max_wait=max_wait), credentials=credentials
        )
        self.mailbox_account = Account(
            self.mailbox_address,
            credentials=credentials,
            autodiscover=True,
            config=self.exchangelib_config,
        )

        # Setup correction folder and done folder
        self.correction_folder_name = correction_folder_name
        self.done_folder_name = done_folder_name

        # Sender email address as mailbox address by default
        self.sender = self.mailbox_address
        self.sender_account = self.mailbox_account

        logger.info(f"Connected to mailbox {self.mailbox_address}")

    @property
    def done_folder_name(self):
        return self._done_folder_name

    @done_folder_name.setter
    def done_folder_name(self, done_folder_name):
        if not done_folder_name:
            self._done_folder_name = None
            self.done_folder = None
        else:
            self._done_folder_name = done_folder_name
            self.done_folder = self.mailbox_account.inbox / done_folder_name

    @property
    def correction_folder_name(self):
        return self._correction_folder_name

    @correction_folder_name.setter
    def correction_folder_name(self, correction_folder_name):
        if not correction_folder_name:
            self._correction_folder_name = None
            self.correction_folder = None
        else:
            self._correction_folder_name = correction_folder_name
            self.correction_folder = self.mailbox_account.inbox / correction_folder_name

    def set_sender_address(self, sender_address, sender_password, max_wait=60):
        """
        Email address to use when sending emails.
        This may differ from the mailbox address.

        Parameters
        ----------
        sender_address: str
        sender_password: str
        max_wait: int
        """
        credentials = Credentials(sender_address, sender_password)
        exchangelib_config = Configuration(
            retry_policy=FaultTolerance(max_wait=max_wait), credentials=credentials
        )
        self.sender_account = Account(
            sender_address,
            credentials=credentials,
            autodiscover=True,
            config=exchangelib_config,
        )
        self.sender = sender_address

    def create_folders(self, folder_list, base_folder_name=None):
        """Create folders in the mailbox.

        Parameters
        ----------
        folder_list : list
            Create folders in the mailbox
        base_folder_name : str
            New folders will be created inside the folder base_folder_name (Defaults to inbox)
        """
        self.folder_list = folder_list

        # Setup base folder
        if base_folder_name:
            base_folder = self.mailbox_account.inbox / base_folder_name
        else:
            base_folder = self.mailbox_account.inbox

        # Check existing folders
        existing_folders = [f.name for f in base_folder.children]

        # Create new folders
        for folder_name in folder_list:
            if folder_name not in existing_folders:
                f = Folder(parent=base_folder, name=folder_name)
                f.save()
                logger.info(
                    f"Created subfolder {folder_name} in folder {base_folder_name}"
                )

    def get_emails(self, max_emails=100, base_folder=None):
        """
        Load emails in the inbox.

        Parameters
        ----------
        max_emails: int
             Maximum number of emails to load
        base_folder: str
            Folder to fetch

        Returns
        -------
        df_new_emails: pandas.DataFrame
            DataFrame containing nex emails
        """
        if base_folder:
            folder = self.mailbox_account.inbox / base_folder
        else:
            folder = self.mailbox_account.inbox

        all_new_data = (
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

        new_emails = [
            self._extract_email_attributes(x)
            for x in all_new_data
            if isinstance(x, Message)
        ]
        df_new_emails = pd.DataFrame(new_emails)

        return df_new_emails

    @staticmethod
    def _extract_email_attributes(email_item):
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
        classified_emails,
        raise_missing_folder_error=False,
        id_column="message_id",
        target_column="target",
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
        target_column: str
            Name of the DataFrame column containing target folder names
        """
        target_folders = classified_emails[target_column].unique().tolist()
        inbox = self.mailbox_account.inbox

        for folder in target_folders:
            try:
                destination_folder = inbox / folder
            except ErrorFolderNotFound:
                if raise_missing_folder_error:
                    logger.exception(f"Mailbox (sub)folder '{folder}' not found")
                    raise
                else:
                    logger.warning(f"Mailbox (sub)folder '{folder}' not found")
                    continue

            mask = classified_emails[target_column] == folder
            mids_to_move = classified_emails[mask][id_column]
            items = inbox.filter(message_id__in=mids_to_move).only("id", "changekey")
            self.mailbox_account.bulk_move(
                ids=items, to_folder=destination_folder, chunk_size=5
            )

    def get_corrections(
        self,
        max_emails=100,
        ignore_list=tuple(),
    ):
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
        if not self.correction_folder_name:
            raise AttributeError(
                "You need to specify a correction_folder_name when instantiating the ExchangeConnector class."
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

    def move_to_done(self, emails_id):
        """
        Once the corrected emails have been processed, they can be moved to a "Done" folder.

        Parameters
        ----------
        emails_id: list
            List of emails IDs to be moved to the done folder.
        """
        if (not self.done_folder_name) or (not self.correction_folder_name):
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
            f"Moved {n_items} corrected emails to the folder {self.done_folder_name}"
        )

    def list_subfolders(self, base_folder=None):
        if base_folder:
            folder = self.mailbox_account.inbox / base_folder
        else:
            folder = self.mailbox_account.inbox

        print([f.name for f in folder.children])

    def send_email(self, to, header, body, attachments):
        m = Message(
            account=self.sender_account,
            subject=header,
            body=HTMLBody(body),
            to_recipients=to
        )
        if attachments:
            for key, value in attachments.items():
                m.attach(FileAttachment(name=key, content=bytes(value, 'utf-8')))

        # Send email
        m.send()
        logger.info(
            f"Email sent from account '{self.sender}'"
        )
