"""
Classes of detectors.

Implemented classes: [ThanksDetector, VacationReplyDetector, ExpeditorDetector,
ReplyDetector, TransferDetector, RecipientsDetector]

"""
from typing import Any, Dict, List, Tuple

from melusine.base import MelusineDetector, MelusineItem, MelusineRegex
from melusine.message import Message
from melusine.regex import (
    EmergencyRegex,
    ReplyRegex,
    ThanksRegex,
    TransferRegex,
    VacationReplyRegex,
)


class ThanksDetector(MelusineDetector):
    """
    Class to detect emails containing only thanks text.

    Ex:
    Merci à vous,
    Cordialement
    """

    # Class constants
    BODY_PART: str = "BODY"
    THANKS_PART: str = "THANKS"
    GREETINGS_PART: str = "GREETINGS"

    # Intermediate columns
    THANKS_TEXT_COL: str = "thanks_text"
    THANKS_PARTS_COL: str = "thanks_parts"
    HAS_BODY: str = "has_body"
    THANKS_MATCH_COL: str = "thanks_match"

    def __init__(
        self,
        messages_column: str = "messages",
        name: str = "thanks",
    ) -> None:
        """
        Attributes initialization.

        Parameters
        ----------
        messages_column: str
            Name of the column containing the messages.

        name: str
            Name of the detector.
        """

        # Input columns
        self.messages_column = messages_column
        input_columns: List[str] = [self.messages_column]

        # Output columns
        self.result_column = f"{name}_result"
        output_columns: List[str] = [self.result_column]

        # Detection regex
        self.thanks_regex: MelusineRegex = ThanksRegex()

        super().__init__(
            name=name,
            input_columns=input_columns,
            output_columns=output_columns,
        )
        self.complex_regex_key: str

    def pre_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Extract text to analyse.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        # Check if a BODY part is present in the last message
        has_body: bool = row[self.messages_column][0].has_tags(
            target_tags={self.BODY_PART}, stop_at={self.GREETINGS_PART}
        )

        # Extract the THANKS part in the last message
        thanks_parts: List[Tuple[str, str]] = row[self.messages_column][0].extract_parts(target_tags={self.THANKS_PART})

        # Compute THANKS text
        if not thanks_parts:
            thanks_text: str = ""
        else:
            thanks_text = "\n".join(x[1] for x in thanks_parts)

        # Save debug data
        if debug_mode:
            debug_dict = {
                self.THANKS_PARTS_COL: thanks_parts,
                self.THANKS_TEXT_COL: thanks_text,
                self.HAS_BODY: has_body,
            }
            row[self.debug_dict_col].update(debug_dict)

        # Create new columns
        row[self.THANKS_TEXT_COL] = thanks_text
        row[self.HAS_BODY] = has_body

        return row

    def detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Use regex to detect thanks.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        debug_info: Dict[str, Any] = {}

        text: str = row[self.THANKS_TEXT_COL]

        detection_data = self.thanks_regex(text)
        detection_result = detection_data[self.thanks_regex.MATCH_RESULT]

        # Save debug data
        if debug_mode:
            debug_info[self.thanks_regex.regex_name] = detection_data
            row[self.debug_dict_col].update(debug_info)

        # Create new columns
        row[self.THANKS_MATCH_COL] = detection_result

        return row

    def post_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Apply final eligibility rules.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """

        # Match on thanks regex & Does not contain a body
        row[self.result_column] = row[self.THANKS_MATCH_COL] and not row[self.HAS_BODY]

        return row


class VacationReplyDetector(MelusineDetector):
    """
    Implement a detector which detects automatic response message like vacation or out of office replies.
    """

    # Class constants
    CONST_TEXT_COL_NAME: str = "vacation_reply_text"
    CONST_DEBUG_TEXT_KEY: str = "text"
    CONST_DEBUG_PARTS_KEY: str = "parts"

    def __init__(
        self,
        name: str,
        messages_column: str = "messages",
    ) -> None:
        """
        Attributes initialization.

        Parameters
        ----------
        name: str
            Detector's name.
        messages_column: str
            Name of the column containing the messages.
        """
        self.messages_column = messages_column

        # Detection regex
        self.vacation_reply_regex: MelusineRegex = VacationReplyRegex()

        # Input columns
        input_columns: List[str] = [messages_column]

        # Output columns
        self.result_column = f"{name}_result"
        output_columns: List[str] = [self.result_column]

        super().__init__(
            name=name,
            input_columns=input_columns,
            output_columns=output_columns,
        )

    def pre_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Extract/prepare the text to analyse.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        # Last message body
        last_message: Message = row[self.messages_column][0]
        body_parts = last_message.extract_last_body()

        if body_parts:
            row[self.CONST_TEXT_COL_NAME] = "\n".join(text for tag, text in body_parts)
        else:
            row[self.CONST_TEXT_COL_NAME] = ""

        # Prepare and save debug data
        if debug_mode:
            debug_dict: Dict[str, Any] = {
                self.CONST_DEBUG_TEXT_KEY: row[self.CONST_TEXT_COL_NAME],
            }
            if self.messages_column:
                debug_dict[self.CONST_DEBUG_PARTS_KEY] = body_parts
            row[self.debug_dict_col].update(debug_dict)

        return row

    def detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Use regex to detect if an email is an automatic response like an Out of office or Vacation reply.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        debug_info: Dict[str, Any] = {}

        text: str = row[self.CONST_TEXT_COL_NAME]

        detection_data = self.vacation_reply_regex(text)
        detection_result = detection_data[self.vacation_reply_regex.MATCH_RESULT]

        # Save debug data
        if debug_mode:
            debug_info[self.vacation_reply_regex.regex_name] = detection_data
            row[self.debug_dict_col].update(debug_info)

        row[self.result_column] = detection_result

        return row

    def post_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Apply final eligibility rule.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        return row


class ReplyDetector(MelusineDetector):
    """
    The ReplyDetector detects if an email is a reply.

    If the header of the email starts with "re", it returns True.
    If not, it returns False.
    """

    # class constant
    CONST_ANALYSED_TEXT_COL: str = "reply_text"

    def __init__(
        self,
        name: str,
        header_column: str = "clean_header",
    ) -> None:
        """
        Attributes initialization.

        Parameters
        ----------
        name: str
            Name given to the detector.
        header_column: [str]
            Name of the column used for the email header.
        """
        # Set instance attributes
        self.header_column = header_column

        # Detection regex
        self.reply_regex: MelusineRegex = ReplyRegex()

        # Input columns
        input_columns: List[str] = [self.header_column]

        # Output columns
        self.result_column = f"{name}_result"
        output_columns: List[str] = [self.result_column]

        super().__init__(
            name=name,
            input_columns=input_columns,
            output_columns=output_columns,
        )

    def pre_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Retrieve text to analyze.

        Log debug information if debug_mode is True.

        Parameters
        ----------
        row: MelusineItem
            Data of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        # Retrieve text to be analysed
        row[self.CONST_ANALYSED_TEXT_COL] = row[self.header_column].lower()

        # Store debug infos
        if debug_mode:
            debug_dict = {
                self.CONST_ANALYSED_TEXT_COL: row[self.CONST_ANALYSED_TEXT_COL],
            }
            row[self.debug_dict_col].update(debug_dict)

        return row

    def detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Check if a header starts with "RE:".

        Parameters
        ----------
        row: MelusineItem
            Data of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        debug_info: Dict[str, Any] = {}

        text: str = row[self.CONST_ANALYSED_TEXT_COL]

        detection_data = self.reply_regex(text)
        detection_result = detection_data[MelusineRegex.MATCH_RESULT]

        # Save debug data
        if debug_mode:
            debug_info[self.reply_regex.regex_name] = detection_data
            row[self.debug_dict_col].update(debug_info)

        row[self.result_column] = detection_result

        return row

    def post_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Apply final eligibility rules.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        # No implementation needed
        return row


class TransferDetector(MelusineDetector):
    """
    The TransferDetector detects if an email is a transfer.
    It returns True if the header starts with "tr:", "fwd:" of if the meta is not empty.
    """

    # class constant
    CONST_ANALYSED_TEXT_COL: str = "reply_text"

    # Debug columns
    CONST_DEBUG_MESSAGE_META: str = "messages[0].meta"

    def __init__(
        self,
        name: str,
        header_column: str = "clean_header",
        messages_column: str = "messages",
    ) -> None:
        """
        Attributes initialization.

        Parameters
        ----------
        name: str
            Name given to the detector.
        header_column: [str]
            Name of the column used for the email header.
        messages_column: [str]
            Name of the column used for the message.
        """
        # Set instance attributes
        self.header_column = header_column
        self.messages_column = messages_column
        self.transfer_regex: MelusineRegex = TransferRegex()

        # Input columns
        input_columns: List[str] = [self.header_column, self.messages_column]

        # Output columns
        self.result_column = f"{name}_result"
        output_columns: List[str] = [self.result_column]

        super().__init__(
            name=name,
            input_columns=input_columns,
            output_columns=output_columns,
        )

    def pre_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Retrieve text to analyze.

        Log debug information if debug_mode is True.

        Parameters
        ----------
        row: MelusineItem
            Data of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        row[self.CONST_ANALYSED_TEXT_COL] = row[self.header_column].lower()

        # Store debug infos
        if debug_mode:
            debug_dict = {
                self.CONST_ANALYSED_TEXT_COL: row[self.CONST_ANALYSED_TEXT_COL],
            }
            row[self.debug_dict_col].update(debug_dict)

        return row

    def detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Check if a header starts with "tr: , fwd:".
        or if the body begins with metadata (e.g, From: , To:, Subject:, etc.)

        Parameters
        ----------
        row: MelusineItem
            Data of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        debug_info: Dict[str, Any] = {}

        text: str = row[self.CONST_ANALYSED_TEXT_COL]
        meta: str = row[self.messages_column][0].meta

        detection_data = self.transfer_regex(text)
        detection_result = detection_data[MelusineRegex.MATCH_RESULT]

        # Save debug data
        if debug_mode:
            debug_info[self.transfer_regex.regex_name] = detection_data
            debug_info[self.CONST_DEBUG_MESSAGE_META] = meta
            row[self.debug_dict_col].update(debug_info)

        row[self.result_column] = detection_result or meta != ""

        return row

    def post_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Apply final eligibility rules.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        # No implementation needed
        return row


class EmergencyDetector(MelusineDetector):
    """
    Implement a detector which detects automatic response message like vacation or out of office replies.
    """

    # Class constants
    CONST_TEXT_COL_NAME: str = "effective_text"
    CONST_DEBUG_TEXT_KEY: str = "text"

    def __init__(
        self,
        name: str,
        header_column: str = "header",
        text_column: str = "det_normalized_last_body",
    ) -> None:
        """
        Attributes initialization.

        Parameters
        ----------
        name: str
            Detector's name.
        header_column: str
            Name of the column containing the text of the email.
        """
        self.header_column = header_column
        self.text_column = text_column

        # Detection regex
        self.regex: MelusineRegex = EmergencyRegex()

        # Input columns
        input_columns: List[str] = [header_column, text_column]

        # Output columns
        self.result_column = f"{name}_result"
        output_columns: List[str] = [self.result_column]

        super().__init__(
            name=name,
            input_columns=input_columns,
            output_columns=output_columns,
        )

    def pre_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Extract/prepare the text to analyse.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        # Last message body
        message_text: str = row[self.text_column]
        header: str = row[self.header_column]

        row[self.CONST_TEXT_COL_NAME] = "\n".join([header, message_text])

        # Prepare and save debug data
        if debug_mode:
            debug_dict: Dict[str, Any] = {
                self.CONST_DEBUG_TEXT_KEY: row[self.CONST_TEXT_COL_NAME],
            }
            row[self.debug_dict_col].update(debug_dict)

        return row

    def detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Apply regex on the effective text.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        debug_info: Dict[str, Any] = {}

        text: str = row[self.CONST_TEXT_COL_NAME]

        detection_data = self.regex(text)
        detection_result = detection_data[self.regex.MATCH_RESULT]

        # Save debug data
        if debug_mode:
            debug_info[self.regex.regex_name] = detection_data
            row[self.debug_dict_col].update(debug_info)

        row[self.result_column] = detection_result

        return row

    def post_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Apply final eligibility rule.

        Parameters
        ----------
        row: MelusineItem
            Content of an email.
        debug_mode: bool
            Debug mode activation flag.

        Returns
        -------
        row: MelusineItem
            Updated row.
        """
        return row
