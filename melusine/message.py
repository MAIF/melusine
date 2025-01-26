"""
Data container class for email.

An email body can contain many "messages".

Implemented classes: [Message]
"""

import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from melusine import config


class Message:
    """
    Class acting as a data container for email data (text, meta and features)
    """

    DEFAULT_STR_LINE_LENGTH = 120
    DEFAULT_STR_TAG_NAME_LENGTH = 22

    def __init__(
        self,
        text: str,
        header: str = "",
        meta: str = "",
        date: Optional[datetime] = None,
        text_from: str = "",
        text_to: Optional[str] = None,
        tags: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Attributes initialization.

        Parameters
        ----------
        text: str
            Message text content.
        header: str
            Message text header.
        meta: str
            Message raw metadata.
        date: datetime
            Message date.
        text_from: str
            Email sender.
        text_to: str
            Email receiver.
        tags: List[Tuple[str, str]]
            Tagged test parts.
            (should be passed as init argument for debug purposes only)
        """
        self.text = text
        self.header = header
        self.meta = meta
        self.date = date
        self.text_from = text_from
        self.text_to = text_to

        self.tags = tags
        self.clean_header: str = ""
        self.clean_text: str = ""

        self.effective_tag_key = "base_tag"
        self.effective_text_key = "base_text"

    @property
    def str_tag_name_length(self) -> int:
        """
        When printing a message, number of characters for the TAG field.
        """
        if "message" not in config:
            return self.DEFAULT_STR_TAG_NAME_LENGTH
        else:
            return config["message"].get("str_tag_name_length", self.DEFAULT_STR_TAG_NAME_LENGTH)

    @property
    def str_line_length(self) -> int:
        """
        When printing a message, total number of characters in each line (text + separation + tag).
        """
        if "message" not in config:
            return self.DEFAULT_STR_LINE_LENGTH
        else:
            return config["message"].get("str_line_length", self.DEFAULT_STR_LINE_LENGTH)

    def extract_parts(
        self,
        target_tags: Optional[Iterable[str]] = None,
        stop_at: Optional[Iterable[str]] = None,
        tag_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Function to extract target tags from the message.

        Parameters
        ----------
        target_tags:
            Tags to be extracted.
        stop_at:
            Tags for which extraction should stop.
        tag_type:
            Type of tags to consider.

        Returns
        -------
        _: List of extracted tags.
        """
        if not self.tags:
            return []

        if tag_type is None:
            tag_type = self.effective_tag_key

        # List of tags in the message
        tag_name_list: List[str] = [x[tag_type] for x in self.tags]

        if target_tags is None:
            target_tags = tag_name_list

        # When stop tags are specified, work on a restricted message
        # (Ex: All tags until GREETINGS)
        if stop_at:
            upper_bound: int = len(tag_name_list)
            for tag_name in stop_at:
                if tag_name in tag_name_list:
                    upper_bound = min(upper_bound, tag_name_list.index(tag_name))
            # Restrict message
            effective_tags = self.tags[:upper_bound]
        else:
            effective_tags = self.tags

        return [x for x in effective_tags if x[tag_type] in target_tags]

    def extract_text(
        self,
        target_tags: Optional[Iterable[str]] = None,
        stop_at: Optional[Iterable[str]] = None,
        tag_type: Optional[str] = None,
        text_type: Optional[str] = None,
        separator: str = "\n",
    ) -> str:
        """
        Function to extract target tags from the message.

        Parameters
        ----------
        target_tags:
            Tags to be extracted.
        stop_at:
            Tags for which extraction should stop.
        tag_type:
            Type of tags to consider.
        text_type:
            Type of text to consider
        separator:
            Separator to join the extracted texts.

        Returns
        -------
        _: List of extracted tags.
        """
        if text_type is None:
            text_type = self.effective_text_key
        parts = self.extract_parts(target_tags=target_tags, stop_at=stop_at, tag_type=tag_type)
        return separator.join([x[text_type] for x in parts])

    def extract_last_body(
        self,
        target_tags: Iterable[str] = ("BODY",),
        stop_at: Iterable[str] = ("GREETINGS",),
        tag_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract the BODY parts of the last message in the email.

        Parameters
        ----------
        target_tags: Iterable[str]
        stop_at: Iterable[str]
        tag_type: Type of tags to consider.

        Returns
        -------
        _: List[Tuple[str, str]]
        """
        return self.extract_parts(target_tags=target_tags, stop_at=stop_at, tag_type=tag_type)

    def has_tags(
        self,
        target_tags: Iterable[str] = ("BODY",),
        stop_at: Optional[Iterable[str]] = None,
        tag_type: Optional[str] = None,
    ) -> bool:
        """
        Function to check if input tags are present in the message.

        Parameters
        ----------
        target_tags:
            Tags of interest.
        stop_at:
            Tags for which extraction should stop.
        tag_type:
            Type of tags to consider.

        Returns
        -------
        _: bool
            True if target tags are present in the message.
        """
        if self.tags is None:
            return False

        if tag_type is None:
            tag_type = self.effective_tag_key

        if not stop_at:
            stop_at = set()

        found: bool = False
        for tag_data in self.tags:
            tag = tag_data[tag_type]

            # Check if tag in tags of interest
            if tag in target_tags:
                found = True
                break

            # Stop when specified tag is reached
            if tag in stop_at:
                break

        return found

    def format_tags(self, tag_type: Optional[str] = None, text_type: Optional[str] = None) -> str:
        """
        Create a pretty formatted representation of text and their associated tags.

        Returns:
            _: Pretty formatted representation of the tags and texts.
        """
        if tag_type is None:
            tag_type = self.effective_tag_key

        if text_type is None:
            text_type = self.effective_text_key

        if self.tags is None:
            return self.text
        else:
            tag_text_length = self.str_line_length - self.str_tag_name_length
            text = ""
            for tag_data in self.tags:
                tag_name = tag_data[tag_type]
                tag_text = tag_data[text_type]
                text += tag_text.ljust(tag_text_length, ".") + tag_name.rjust(self.str_tag_name_length, ".") + "\n"

        return text.strip()

    def __repr__(self) -> str:
        """
        String representation.

        Returns
        -------
        _: str
            Readable representation of the Message.
        """
        if self.meta:
            meta = re.sub(r"\n+", r"\n", self.meta).strip("\n ")
        else:
            meta = "NA"
        text: str = re.sub(r"\n+", r"\n", self.text)
        return f"Message(meta={repr(meta)}, text={repr(text)})"

    def __str__(self) -> str:
        """
        Repr representation.

        Returns
        -------
        _: str
            Readable representation of the Message.
        """
        title_len = 22
        fill_len = (self.str_line_length - title_len) // 2

        text = ""
        text += f"{'='*fill_len}{'Message':^{title_len}}{'='*fill_len}\n"
        text += f"{'-'*fill_len}{'Meta':^{title_len}}{'-'*fill_len}\n"
        text += f"{self.meta or 'N/A'}\n"
        text += f"{'-'*fill_len}{'Text':^{title_len}}{'-'*fill_len}\n"
        text += self.format_tags() + "\n"
        text += f"{'='*fill_len}{'=' * title_len}{'='*fill_len}\n\n"

        return text
