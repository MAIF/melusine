"""
Define Processors
Processors are objects that can be used as standalone or as steps of a MelusinePipeline.

Implemented classes: [
    Normalizer,
    RegexTokenizer,
    Phraser,
    BaseSegmenter,
    Segmenter,
    BaseExtractor,
    TextExtractor,
    TokensExtractor,
    Tag,
    BaseContentTagger,
    ContentTagger,
    DeterministicTextFlagger,
    Cleaner,
]
"""
from __future__ import annotations

import logging
import re
import unicodedata
from abc import abstractmethod
from re import Pattern
from typing import Any, Iterable, Sequence, Union

import arrow

from melusine.base import MelusineDataset, MelusineTransformer
from melusine.message import Message

logger = logging.getLogger(__name__)


class Normalizer(MelusineTransformer):
    """
    Normalizer transforms raw text into standard text by:
    - Lowercasing
    - Applying unicode normalization standards such as NFD or NFKD
    """

    def __init__(
        self,
        form: str = "NFKD",
        lowercase: bool = True,
        fix_newlines: bool = True,
        input_columns: str = "text",
        output_columns: str = "text",
    ):
        """
        Parameters
        ----------
        form: str
            Unicode normalization form
        lowercase: bool
            If True, lowercase the text
        input_columns: str
            Input columns for the transform operation
        output_columns: str
            Outputs columns for the transform operation
        """

        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
        )

        if "messages" in self.input_columns:
            self.func = self.normalize_message
        else:
            self.func = self.normalize_text

        # Unicode normalization form
        self.form = form

        # Lower casing
        self.lowercase = lowercase

        # Fix newlines
        self.fix_newlines = fix_newlines

    def normalize_message(self, message_list: list[Message]) -> list[Message]:
        """
        Normalize the text of a message.

        Parameters
        ----------
        message_list: list[Message]
            Input message list

        Returns
        -------
        _: list[Message]
            Normalized message list
        """

        for message in message_list:
            message.clean_text = self.normalize_text(message.text)
            message.clean_header = self.normalize_text(message.header)

        return message_list

    def normalize_text(self, text: str) -> str:
        """
        Apply the normalization transformations to the text.

        Parameters
        ----------
        text: str
            Input text to be normalized

        Returns
        -------
        text: str
            Normalized text
        """
        if not isinstance(text, str):
            text = ""

        # Uncommon characters
        text = text.replace("’", "'")
        text = text.replace("œ", "oe")

        # Unicode normalization
        if self.form:
            text = unicodedata.normalize(self.form, text).encode("ASCII", "ignore").decode("utf-8")

        # Lowercasing
        if self.lowercase:
            text = text.lower()

        # Fix newlines
        if self.fix_newlines:
            # Replace platform newlines by standard newline
            text = "\n".join(text.splitlines())

            # Replace multipe spaces by single space
            text = re.sub(r" +", " ", text)

            # Replace multipe newline/spaces patterns by single newline
            text = re.sub(r" *\n+ *", "\n", text)

            # Undesired newlines following quotes
            # Ex : "<\nabc@domain.fr\n>" => "<abc@domain.fr>"
            text = re.sub(r"<\n(\w)", r"<\1", text)
            text = re.sub(r"(\w)\n>", r"\1>", text)

        # Replace non-breaking spaces
        text = text.replace("\xa0", " ")

        return text


class RegexTokenizer(MelusineTransformer):
    """
    Class to split a text into tokens using a regular expression.
    """

    def __init__(
        self,
        tokenizer_regex: str = r"\w+(?:[\?\-\"_]\w+)*",
        stopwords: list[str] | None = None,
        lowercase: bool = True,
        normalization_form: str | None = None,
        input_columns: str = "text",
        output_columns: str = "tokens",
    ):
        """
        Parameters
        ----------
        tokenizer_regex: str
            Regex used to split the text into tokens
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.tokenize,
        )
        # Lowercasing
        self.lowercase = lowercase

        # Normalization
        self.normalization_form = normalization_form

        # Tokenizer regex
        self.tokenizer_regex = tokenizer_regex

        # Stopwords
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

    def _text_to_tokens(self, text: str) -> Sequence[str]:
        """
        Method to split a text into a list of tokens.

        Parameters
        ----------
        text: str
            Text to be split

        Returns
        -------
        tokens: Sequence[str]
            List of tokens
        """
        tokens = re.findall(self.tokenizer_regex, text, re.M + re.DOTALL)

        return tokens

    def _remove_stopwords(self, tokens: Sequence[str]) -> Sequence[str]:
        """
        Method to remove stopwords from tokens.

        Parameters
        ----------
        tokens: Sequence[str]
            List of tokens

        Returns
        -------
        tokens: Sequence[str]
            List of tokens without stopwords
        """
        return [token for token in tokens if token not in self.stopwords]

    def tokenize(self, text: str) -> Sequence[str]:
        """
        Method to apply the full tokenization pipeline on a text.

        Parameters
        ----------
        text: str
            Input text to be tokenized

        Returns
        -------
        tokens: Sequence[str]
            List of tokens
        """
        # Lowercase
        if self.lowercase:
            text = text.lower()

        if self.normalization_form:
            text = unicodedata.normalize(self.normalization_form, text).encode("ASCII", "ignore").decode("utf-8")

        # Text splitting
        tokens = self._text_to_tokens(text)

        # Stopwords removal
        tokens = self._remove_stopwords(tokens)

        return tokens


class BaseSegmenter(MelusineTransformer):
    """
    Class to split a conversation into a list of Messages.
    This is an abstract class defining the Segmenter interface.
    Melusine users should implement a subclass or use an existing subclass to segment their emails.
    """

    def __init__(
        self,
        strip_characters: str = "\n >-",
        input_columns: str = "body",
        output_columns: str = "messages",
        regex_flags: re.RegexFlag = re.MULTILINE | re.IGNORECASE,
    ):
        """
        Parameters
        ----------
        strip_characters: str
            Characters to be stripped of text segments
        input_columns: str
            Input columns for the transform operation
        output_columns: str
            Outputs columns for the transform operation
        regex_flags: re.RegexFlag
            Regex flags for segmentation
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.segment_text,
        )

        self.strip_characters = strip_characters

        # Compile segmentation regex
        regex_list = self.create_segmentation_regex_list()
        self._compiled_segmentation_regex = self.compile_regex_from_list(regex_list, flags=regex_flags)

    @staticmethod
    @abstractmethod
    def create_segmentation_regex_list() -> Iterable[str]:
        """
        Method to create a compiled regex that can be used to segment an email.

        Returns
        -------
        _: Iterable[str]
            List of segmentation regexs
        """

    @staticmethod
    def compile_regex_from_list(regex_list: Iterable[str], flags: int | re.RegexFlag = re.M) -> Pattern:
        """
        Method to create a meta-regex from a list of regexs.

        Parameters
        ----------
        regex_list: Iterable[str]
            List of individual regexs
        flags:  int | RegexFlag
            Regex flags

        Returns
        -------
        _:  Pattern[AnyStr]
            Compiled meta-regex
        """
        regex_list = ["(?:" + r + ")" for r in regex_list]
        regex = "|".join(regex_list)

        # Add an overall capture group
        regex = "(" + regex + ")"

        return re.compile(regex, flags=flags)

    def create_messages(self, match_list: list[str]) -> list[Message]:
        """
        Method to create Message instances based on the segmented email data.

        Parameters
        ----------
        match_list: list[str]
            List of text elements matched by the segmentation regex

        Returns
        -------
        _: list[Message]
        """
        # Create first message meta based on email meta
        first_message_meta = ""

        # Strip characters
        match_list = [x.strip(self.strip_characters) for x in match_list]

        # Case email starts with a transition pattern
        if match_list[0] == "":
            if len(match_list) > 1:
                # Adapt first message meta
                first_message_meta = match_list[1]

                # Skip first 2 indices (1st text + 1st meta pattern)
                match_list = match_list[2:]

            else:
                # Empty message
                return [Message(text="")]

        # Insert placeholder for the first message meta
        match_list.insert(0, "")

        # Even indices are meta patterns
        meta_list = [x for i, x in enumerate(match_list) if i % 2 == 0]

        # Odd indices are text
        text_list = [x for i, x in enumerate(match_list) if i % 2 == 1]

        # Replace first message meta
        meta_list[0] = first_message_meta

        return [Message(text=text, meta=meta) for text, meta in zip(text_list, meta_list)]

    def segment_text(self, text: str) -> list[Message]:
        """
        Method to segment a conversation by splitting the text on  transition patterns.
        Ex:
        > Input : "Thank you\nSent by Mr Smith\nHello\nSee attached the document.\nBest Regards"
        > Output :
            - Message(text="Thank you", meta=None)
            - Message(text="Hello\nSee attached the document.\nBest Regards", meta="Sent by Mr Smith")

        Parameters
        ----------
        text: str
            Full text conversation

        Returns
        -------
        _: list[Message]
            List of messages
        """
        # Strip start / end characters
        text = text.strip(self.strip_characters)

        # Split text using the compiled segmentation regex
        matches = self._compiled_segmentation_regex.split(text)

        return self.create_messages(matches)


class Segmenter(BaseSegmenter):
    """
    Class to split a conversation into a list of Messages.
    Inherits from BaseSegmenter.
    Implement methods to segment french emails.
    """

    def __init__(
        self,
        strip_characters: str = "\n >-",
        input_columns: str = "body",
        output_columns: str = "messages",
    ):
        """
        Parameters
        ----------
        strip_characters: str
            Characters to be stripped of text segments
        input_columns: str
            Input columns for the transform operation
        output_columns: str
            Outputs columns for the transform operation
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            strip_characters=strip_characters,
        )

    @staticmethod
    def create_segmentation_regex_list() -> Iterable[str]:
        """
        Method to create a compiled regex that can be used to segment an email.

        Returns
        -------
        _: Iterable[str]
            List of segmentation regexs
        """
        # Meta patterns of the form "META_KEYWORD : META_CONTENT"
        # Ex: "De : jean@gmail.com"
        meta_keywords_list_with_semicolon = [
            r"Date",
            r"De",
            r"Exp[ée]diteur",
            r"[ÀA]",
            r"Destinataire",
            r"Envoy[ée](?: le| par)?",
            r"R[ée]pondre [àa]",
            r"Pour",
            r"From",
            r"To",
            r"Sent",
            r"Cc",
        ]
        piped_keywords_with_semicolon = "(?:" + "|".join(meta_keywords_list_with_semicolon) + ")"
        starter_pattern_with_semicolon = rf"^.{{,5}}(?:{piped_keywords_with_semicolon} ?\n? ?: *\n?)"

        # Meta patterns of the form "META_KEYWORD"
        # Ex: "Transféré par jean@gmail.com"
        # (pas de ":")
        # Ex: "------ Message transmis ------"

        regex_weekdays = (
            r"(?:[Ll]undi|[Ll]un\.|[Mm]ardi|[Mm]ar\.|[Mm]ercredi|[Mm]er\.|[Jj]eudi|[Jj]eu\.|"  # noqa
            r"[Vv]endredi|[Vv]en\.|[Ss]amedi|[Ss]am\.|[Dd]imanche|[Dd]im\.)"  # noqa
        )
        regex_months = (
            r"(?:[Jj]anvier|[Ff][ée]vrier|[Mm]ars|[Aa]vril|[Mm]ai|[Jj]uin|[Jj]uillet|"  # noqa
            r"[Aa]o[ûu]t|[Ss]eptembre|[Oo]ctobre|[Nn]ovembre|[Dd][eé]cembre|"  # noqa
            r"(?:janv?|f[ée]vr?|mar|avr|juil?|sept?|oct|nov|d[ée]c)\.)"
        )

        meta_keywords_list_without_semicolon = [
            # Le 2021-01-02 11:20 jane@gmail.fr a écrit :
            # Le 02 juillet 1991 à 11:20 jane@gmail.fr a écrit :
            # Le mardi 31 août 2021 à 11:09, <ville@maif.fr> a écrit :
            (
                rf"\bLe (?:"
                rf"\d{{2}}/\d{{2}}/\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}}|{regex_weekdays}|"  # noqa
                rf"\d{{1,2}} {regex_months})(?:.|\n){{,30}}\d{{2}}:\d{{2}}(?:.|\n){{,50}}(?:\<.{{,30}}\>.{{,5}})?\ba [éecrit]"  # noqa
            ),
            r"Transf[ée]r[ée] par",
            r"D[ée]but du message transf[ée]r[ée] :",
            r"D[ée]but du message r[ée]exp[ée]di[ée] :",
            r"Message transmis",
            r"(?:Message|[Mm]ail) transf[ée]r[ée]",
            r"(?:Courriel|Message|Mail) original",
            r"(?:Message|Mail|Courriel) d'origine",
            r"Original [Mm]essage",
            r"Forwarded message",
            r"Forwarded by",
        ]
        piped_keywords_without_semicolon = "(?:" + "|".join(meta_keywords_list_without_semicolon) + ")"  # noqa
        starter_pattern_without_semicolon = f"{piped_keywords_without_semicolon}(?:[\n ]*--+)?"

        # Combine pattern with and without semicolon
        starter_pattern = rf"(?:{starter_pattern_with_semicolon}|{starter_pattern_without_semicolon})"  # noqa

        # Match everything until the end of the line.
        # Match End of line "\n" and "space" characters
        end_pattern = r".*[\n ]*"

        # Object / Subject pattern (These patterns are not sufficient to trigger segmentation)
        object_line_pattern = "(?:^.{,5}(?:Objet|Subject|Sujet) ?\n? ?: *\n?)" + end_pattern

        # Starters are separated from the meta-values by
        # a semicolon and optional spaces/line breaks
        full_generic_meta_pattern = rf"(?:{starter_pattern}{end_pattern}(?:{object_line_pattern})*)+"

        # Make a tuple of patterns
        pattern_list = (full_generic_meta_pattern,)

        return pattern_list


class BaseExtractor(MelusineTransformer):
    """
    Class to extract data from a list of messages.
    This is an abstract class defining the interface for extractor classes.
    Melusine users should implement a subclass (or use an existing subclass) to extract data from a list of messages.
    """

    def __init__(
        self,
        input_columns: str | Iterable[str],
        output_columns: str,
    ):
        """
        Parameters
        ----------
        input_columns: Union[str, Iterable[str]]
            Input columns for the transform operation
        output_columns: str
            Outputs columns for the transform operation
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.extract,
        )

    @abstractmethod
    def extract(self, message_list: list[Message]) -> Any:
        """
        Method to extract data from a list of messages.

        Parameters
        ----------
        message_list: list[Message]
            List of Messages

        Returns
        -------
        _: str
            Extracted text
        """


class TextExtractor(BaseExtractor):
    """
    Class to extract text data from a list of messages.
    """

    def __init__(
        self,
        input_columns: str = "messages",
        output_columns: str = "last_message",
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        sep: str = "\n",
        n_messages: int | None = 1,
        stop_at: Iterable[str] = ("GREETINGS",),
    ):
        """
        Parameters
        ----------
        input_columns: str
            Input columns for the transform operation
        output_columns: str
            Outputs columns for the transform operation
        include_tags: list[str]
            Message tags to be included in the text extraction
        exclude_tags: list[str]
            Message tags to be excluded from the text extraction
        sep: str
            Separation symbol to join text parts
        n_messages: Union[int, None]
            Number of messages to take into account (starting with the latest)
        stop_at: list[str]
            When stop_at tags are encountered, stop extracting text of the message
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
        )

        if include_tags and exclude_tags:
            raise ValueError(f"{type(self).__name__} :" "You should specify only of include_tags/exclude_tags")

        self.include_tags = include_tags
        self.exclude_tags = exclude_tags
        self.sep = sep
        self.n_messages = n_messages
        self.stop_at = stop_at

    def extract(self, message_list: list[Message]) -> str:
        """
        Method to extract text parts from a list of messages.

        Parameters
        ----------
        message_list: list[Message]
        Input message list

        Returns
        -------
        _: str
            Extracted text
        """
        if self.n_messages is None:
            n_messages = len(message_list)
        else:
            n_messages = self.n_messages

        text_list = list()

        for message in message_list[:n_messages]:
            # Message has been tagged
            if message.tags is not None:
                if self.include_tags:
                    tags = message.extract_parts(target_tags=self.include_tags, stop_at=self.stop_at)
                    message_text_list = [x[1] for x in tags]
                elif self.exclude_tags:
                    tags = message.extract_parts(target_tags=None, stop_at=self.stop_at)
                    message_text_list = [part for tag, part in tags if tag not in self.exclude_tags]
                else:
                    message_text_list = [part for tag, part in message.tags]

                # Join message text list
                extracted_text = self.sep.join(message_text_list)

            # Message has not been tagged
            else:
                extracted_text = message.text

            text_list.append(extracted_text)

        return self.sep.join(text_list).strip()


class TokensExtractor(BaseExtractor):
    """
    Class to extract tokens from different DataFrame columns.
    Ex:
    > (input column 1) body_tokens: ["hello", "how", "are", "you"]
    > (input column 2) header_tokens: ["catch", "up"]
    > (output column) all_tokens: ["catch", "up", "hello", "how", "are", "you"]

    """

    def __init__(
        self,
        input_columns: str | Iterable[str] = ("header_tokens", "body_tokens"),
        output_columns: str = "tokens",
        sep_token: str = "[PAD]",
        pad_size: int = 5,
    ):
        """
        Parameters
        ----------
        input_columns: Union[str, Iterable[str]]
            Input columns for the transform operation
        output_columns: str
            Outputs columns for the transform operation
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
        )

        self.sep_token = sep_token
        self.pad_size = pad_size

    def extract(self, row: MelusineDataset) -> list[str]:
        """
        Method to extract tokens from different columns of a DataFrame.

        Parameters
        ----------
        row: MelusineDataset
            Emails input data

        Returns
        -------
        _: list[str]
            List of extracted tokens
        """

        pad_pattern = [self.sep_token] * self.pad_size
        tokens = list()
        for col in self.input_columns:
            tokens += row[col]
            tokens += pad_pattern

        # Remove trailing padding tokens
        tokens = tokens[: -self.pad_size]

        return tokens


class Tag(property):
    """
    Class used by the ContentTagger to identify text tags such as:
    - BODY
    - HELLO
    - GREETINGS
    """


TagPattern = Union[str, Iterable[str], re.Pattern]


class BaseContentTagger(MelusineTransformer):
    """
    Class to add tags to a text
    This is an abstract class defining the interface for all ContentTaggers.
    Melusine users should implement a subclass (or use an existing subclass) to add tags to texts.
    """

    def __init__(
        self,
        input_columns: str = "messages",
        output_columns: str = "messages",
        tag_list: list[str] | None = None,
        default_tag: str = "BODY",
        valid_part_regex: str = r"[a-z0-9?]",
        default_regex_flag: int = re.IGNORECASE,
        post_process: bool = True,
        text_attribute: str = "text",
    ):
        """
        Parameters
        ----------
        input_columns: str
        output_columns: str
        tag_list: list[str]
            (Ordered) List of tags to look for
        default_tag: str
            Tag given to arbitrary text parts
        default_regex_flag: int
            Default flag to compile regex
        text_attribute: str
            Message attribute containing the text data
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.tag_email,
        )

        # If no tag list is specified, use all the tags
        if not tag_list:
            self.tag_list = self.get_tag_list()
        else:
            self.tag_list = tag_list

        # Sep default tag
        self.default_tag = default_tag

        # Set default flag
        self.default_regex_flag = default_regex_flag

        # Set text attribute
        self.text_attribute = text_attribute

        # Activate post-processing
        self.post_process = post_process

        # Pattern to split text into sentences (=parts)
        self.split_pattern = self.compile_split_pattern()

        # Pattern to validate that a text part is valid
        self.valid_part_regex = valid_part_regex

        # Build the regex_dict
        self.regex_dict = {}
        for tag in self.tag_list:
            self.regex_dict[tag] = self.compile_tag_regex(tag)

    def __getitem__(self, key: str) -> re.Pattern:
        """
        Method to access regex corresponding to individual tags easily.
        Ex:
        > t = ContentTagger()
        > t["HELLO"].match("bonjour")

        Parameters
        ----------
        key: str
            Name of a tag

        Returns
        -------
        _: re.Pattern
            Compiled regex
        """
        return self.regex_dict[key]

    @staticmethod
    def compile_split_pattern() -> re.Pattern:
        """
        Method to compile the sentence split regex pattern.

        Ex:
        Bonjour Mr. Dupont. Salutations
        will be splited using the split pattern into
        ["Bonjour Mr. Dupont", "Salutations"]

        Returns
        -------
        _: re.Pattern
            Compiled regex
        """
        # Dot exception patterns
        _madame_pattern = r"(?<!\bM[Mm]e)"
        _monsieur_pattern = r"(?<!\bMr)"
        _mademoiselle_pattern = r"(?<!\bMelle)"
        _mademoiselle_pattern2 = r"(?<!\bMlle)"
        _doctor_pattern = r"(?<!\bDr)"
        _professor_pattern = r"(?<!\bPr)"
        _acronym_pattern = r"(?<!\b[A-Z])"

        # Hello exception patterns
        hello_split_pattern = r"(?:^[Bb](?:onjour|onsoir)\s*,)+"

        # Regular_split_patterns
        newline_pattern = r"\n+"
        dot_pattern = r"(?:[.?!]+(?![a-zA-Z0-9]))+"

        # Build pattern to avoid undesired splitting on the point character
        dot_pattern_with_negativelookbehind = (
            r"(?:"
            + _madame_pattern
            + _monsieur_pattern
            + _mademoiselle_pattern
            + _mademoiselle_pattern2
            + _doctor_pattern
            + _professor_pattern
            + _acronym_pattern
            + dot_pattern
            + r")"
        )

        sentence_split_pattern = (
            r"( *(?:" + f"{dot_pattern_with_negativelookbehind}|{hello_split_pattern}|{newline_pattern}" + r")+) *"
        )
        return re.compile(sentence_split_pattern)

    @classmethod
    def get_tag_list(cls) -> list[str]:
        """
        Method to get the list of available tags.

        Returns
        -------
        _: list[str]
            List of tags
        """
        return [p for p in dir(cls) if isinstance(getattr(cls, p), Tag)]

    def tag_email(self, messages: list[Message]) -> list[Message] | None:
        """
        Method to apply content tagging on an email (= List of Messages)

        Parameters
        ----------
        messages : list[Message]
            List of messages

        Returns
        -------
        messages : list[Message]
            List of messages after content tagging
        """
        if not messages:
            return None

        for message in messages:
            tags = self.tag_text(getattr(message, self.text_attribute))
            message.tags = tags

        return messages

    def compile_tag_regex(self, tag: str) -> re.Pattern:
        """
        Method to validate and compile the regex associated with the input tag.
        Return an error if the regex pattern is malformed.

        Parameters
        ----------
        tag: str
            Tag of interest

        Returns
        -------
        _: re.Pattern
            compiled regex
        """
        # Collect data associated with the input tag
        if hasattr(self, tag):
            regex = getattr(self, tag)
        else:
            raise ValueError(f"Unknown tag {tag}")

        # If a list is provided, pipe it into a string
        if (not isinstance(regex, str)) and isinstance(regex, Iterable):
            regex = "|".join(regex)

        # Compile regex from string
        if isinstance(regex, str):
            try:
                regex = re.compile(regex, flags=self.default_regex_flag)
            except re.error:
                raise ValueError(f"Invalid regex for tag {tag}:\n{regex}")
        elif isinstance(regex, re.Pattern):
            pass
        else:
            raise ValueError(
                f"Tag {tag} does not return any of the supported types : "
                "str "
                "list[str] "
                "re.Pattern "
                f"Got {type(regex)} instead."
            )

        return regex

    def tag_text(self, text: str) -> list[tuple[str, str]]:
        """
        Method to apply content tagging on a text.

        Parameters
        ----------
        text: str
            Input text

        Returns
        -------
        _: list[tuple[str, str]]
            List of tag/text couples (ex: [("HELLO", "bonjour")])
        """
        parts = self.split_text(text)
        tags = list()
        for part in parts:
            tags.append(self.tag_part(part))

        # Post process tags
        if self.post_process:
            tags = self.post_process_tags(tags)

        return tags

    def split_text(self, text: str) -> list[str]:
        """
        Method to split input text into sentences/parts using a regex.

        Parameters
        ----------
        text: str
            Input text

        Returns
        -------
        _: list[str]
            List of parts/sentences
        """
        # Replace multiple spaces by single spaces
        text = re.sub(r" +", " ", text)

        # Split text into sentences
        parts = self.split_pattern.split(text)

        # Cleanup sentence split
        clean_parts = self.clean_up_after_split(parts)

        return [p.strip() for p in clean_parts if self.validate_part(p)]

    def validate_part(self, text: str) -> bool:
        """
        Method to validate a text part.
        By default, check that it contains at least one of:
        - a letter
        - a number
        - an interrogation mark.

        Parameters
        ----------
        text: Text part to be validated

        Returns
        -------
        _: bool
            True if text part is valid
        """
        return bool(re.search(self.valid_part_regex, text, flags=re.I))

    @staticmethod
    def clean_up_after_split(parts: list[str | None]) -> list[str]:
        """
        Clean up sentences after splitting.
        Typically, put punctuation back at the end of sentences.

        Parameters
        ----------
        parts: list[Union[str, None]]

        Returns
        -------
        clean_parts: list[str]
        """
        clean_parts: list[str] = []
        for part in parts:
            if not part:
                continue

            # Part contains punctuation only
            if (len(clean_parts) > 0) and re.search(r"^[ .?!\n]+$", part):
                # Add characters to the previous part
                clean_parts[-1] += part
                continue

            # Regular part
            clean_parts.append(part.strip("\n"))

        return clean_parts

    def tag_part(self, part: str) -> tuple[str, str]:
        """
        Method to apply tagging on a text chunk (sentence/part).

        Parameters
        ----------
        part: str
            Text chunk

        Returns
        -------
        match_tag: str
            Output tag
        part: str
            Original text
        """
        match_tag = self.default_tag

        for tag, regex in self.regex_dict.items():
            match = regex.match(part)
            if match:
                match_tag = tag
                break

        return match_tag, part

    @staticmethod
    def word_block(n_words: int, word_character_only: bool = False) -> str:
        """
        Method to dynamically generate regex patterns to match block of words.

        Parameters
        ----------
        n_words: int
            Number of words to be matched
        word_character_only: bool
            If True, match word characters only

        Returns
        -------
        _: str
            Regex matching desired pattern
        """
        if word_character_only:
            positive = r"\w"
        else:
            # Non-space characters except - and – (considered word separators)
            positive = r"[^\r\s\t\f\v \-–]"

        return rf"(?:[ \-–]*(?:{positive}+(?:[ \-–]+{positive}+){{,{n_words - 1}}})? *)"

    def __call__(self, text: str) -> list[tuple[str, str, str]]:
        """
        Method to find all regex patterns matching the input text.

        Parameters
        ----------
        text: str
            Text to match

        Returns
        -------
        match_list: list[tuple[str, str]]
            List of matching regexes and associated tags
        """
        full_match_list = list()

        # Split parts of the input string
        parts = self.split_text(text)
        for part in parts:
            for tag in self.regex_dict.keys():
                # Get regex or list of regexes
                regex = getattr(self, tag)

                # Find matching regexes
                matching_regex_list = self.find_matching_regex_patterns(part, regex)

                # Format result
                full_match_list.extend([(part, tag, regex) for regex in matching_regex_list])

        return full_match_list

    def find_matching_regex_patterns(self, part: str, regex: TagPattern) -> list[str]:
        """
        Given a regex string, a regex pattern or a list of regexes.
        Find all matching patterns
        """
        matching_regex_list = []
        # Tag defined with a string regex
        if isinstance(regex, str):
            regex_match = re.match(regex, part, flags=self.default_regex_flag)
            if regex_match:
                matching_regex_list.append(regex)

        # Tag defined with a list of string regexes
        elif isinstance(regex, Iterable):
            for r in regex:
                regex_match = re.match(r, part, flags=self.default_regex_flag)
                if regex_match:
                    matching_regex_list.append(r)

        else:
            regex_match = regex.match(part)
            if regex_match:
                matching_regex_list.append(regex.pattern)

        return matching_regex_list

    @abstractmethod
    def post_process_tags(self, tags: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        Method to apply tagging rules posterior to the standard regex tagging.

        Parameters
        ----------
        tags: list[tuple[str, str]]
            Original tags

        Returns
        -------
        _: list[tuple[str, str]]
            Post-processed tags
        """


class ContentTagger(BaseContentTagger):
    """
    Class to add tags to a text.
    This class inherits from the base class BaseContentTagger.
    This class implements Content Tagging for French emails.

    Implemented tags:
    - HELLO
    - PJ
    - DISCLAIMER
    - FOOTER
    - GREETINGS
    - SIGNATURE
    """

    ENGLISH_TIMES = ["day", "morning", "afternoon", "evening", "night", "week(-?end)?"]

    def __init__(
        self,
        input_columns: str = "messages",
        output_columns: str = "messages",
        tag_list: list[str] | None = None,
        default_tag: str = "BODY",
        valid_part_regex: str = r"[a-z0-9?]",
        default_regex_flag: int = re.IGNORECASE | re.MULTILINE,
        post_process: bool = True,
        text_attribute: str = "text",
    ):
        """
        Parameters
        ----------
        input_columns: str
        output_columns: str
        tag_list: list[str]
            (Ordered) List of tags to look for
        default_tag: str
            Tag given to arbitrary text parts
        default_regex_flag: int
            Default flag to compile regex
        text_attribute: str
            Message attribute containing the text data
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            tag_list=tag_list,
            default_tag=default_tag,
            valid_part_regex=valid_part_regex,
            default_regex_flag=default_regex_flag,
            post_process=post_process,
            text_attribute=text_attribute,
        )

    @Tag
    def GREETINGS(self) -> str | list[str] | re.Pattern:
        """
        Tag associated with email closure sentences.
        Watchout, this tag typically marks the end of a message.
        Ex: "Cordialement"
        """
        english_times_pattern = "|".join(self.ENGLISH_TIMES)
        return [
            r"^.{0,30}cordialement.{0,30}$",
            r"^.{0,5}sinc[èe]rement.{0,30}$",
            r"^.{0,10}cdl?t.{0,16}$",
            r"^.{0,10}bien aimablement.{0,16}$",
            r"^.{0,10}courtoisement.{0,16}$",
            r"^.{0,10}bien [àa] (?:toi|vous).{0,16}$",
            r"^.{0,10}sentiments? (?:d[ée]vou[ée]s?|mutualistes?).{0,16}$",
            r"^.{0,10}(Veuillez.{,3})?(accepte[zr] .{,8}|receve[zr] .{,8})?(meilleure?s?|sinc[eè]res?|cordiale?s?)? ?(salutations?|sentiments?).{0,16}$",
            r"^.{0,45}(?:(?:l')?expression|assurance) de (?:nos|mes) sentiments.{0,16}$",
            r"^.{0,50}(?:salutations?|sentiments?) distingu.{0,30}$",
            r"^.{0,10}Respectueusement.{0,16}$",
            r"^.{0,20}(?:souhait.{,10})?(?:continuation|r[ée]ception).{0,16}$",
            r"^.{0,20}dans l'attente de (?:votre (?:retour|r[ée]ponse)|vous lire).{0,16}$",
            r"^.{0,30}(?:une )bonne r[ée]ception.{0,16}$",
            r"^.{0,3}Bonne r[ée]ception.{0,3}$",
            r"^.{0,3}votre bien d[ée]vou[ée]e?.{0,3}$",
            r"^.{0,3}amicalement votre.{0,3}$",
            r"^.{,3}je vous prie de croire.{,50}(expression|assurance)?.{,50}(consideration|salutations|sentiments).{,30}$",
            # English
            r"^.{0,3}regards.{0,3}$",
            r"^.{0,3}(best|warm|kind|my) *(regards|wishes)?.{0,3}$",
            r"^.{0,3}(yours)? *(truly|sincere?ly|respectfull?y|faithfully).{0,3}$",
            r"^.{0,3}yours.{0,3}$",
            r"^.{0,3}cheers.{0,3}$",
            "^.{0,3}(talk|write|see you|speak to you) soon.{0,3}$",
            "^.{0,3}take care.{0,3}$",
            "^.{0,3}catch you later.{0,3}$",
            rf"^.{{0,3}}have an? (blessed|excellent|good|fantastic|great) ({english_times_pattern}).{{0,3}}$",
            r"i am looking forward to hearing from you.{0,3}$",
            "^.{0,3}looking forward to your reply.{0,3}$",
            "^.{0,3}hoping to hear from you( soon)?.{0,3}$",
        ]

    @Tag
    def HELLO(self) -> str | list[str] | re.Pattern:
        """
        Tag associated with email opening sentences.
        Sentences that can be either opening or closing should be placed here.
        Ex1: "Bonjour"
        Ex2: "Bonne année"
        """
        english_times_pattern = "|".join(self.ENGLISH_TIMES)

        # === Souhaits de bonheur ===
        # J'espère que vous avez passé un bon week-end, etc
        # Bonne semaine
        bon_list = [
            r"ann[ée]e",
            r"(fin de )?journ[ée]e?",
            r"soir[ée]e?",
            r"week[ -]?end",
            r"we",
            r"nuit",
            r"(fin de )?semaine",
            r"f[eê]tes?",
            r"(fin d')?apr[eè]s[ -]?midi",
        ]

        deb_bon_list = [
            r"Bon(?:ne)?",
            r"Bel(?:le)?",
            r"Beau",
            r"joyeux?(?:se)?",
            r"Excel?lent(?:e)?",
        ]

        hello_words_list = [
            r"ch[èe]r[.()es]{,4}",
            r"bonjour",
            r"bonsoir",
            r"madame",
            r"monsieur",
            r"mesdames",
            r"messieurs",
            # English
            rf"good {english_times_pattern}",
            r"hi( there)?",
            r"hello",
            r"greetings",
            r"dear",
            r"dear (m(rs?|s)\.?|miss|madam|mister|sir)( or (m(rs?|s)\.?|miss|madam|mister|sir))?",
            r"sir",
            r"how are you (doing|today)",
            r"(it is|it's)? ?(good|great) to hear from you",
            r"i hope (you are|you're)( doing)? well",
            rf"i hope (you are|you're) having an? ?(great|wonderful|fantastic)? ({english_times_pattern})",
            r"i hope this email finds you well",
            r"to whom it may concern",
        ]
        hello_pattern = "|".join(hello_words_list)

        return [
            r"^.{0,10}((\b" + hello_pattern + r")\b\s*){1,3}(\w+\b\s*){,4}.{,3}(?!.)$",
            rf"^.{{0,16}}(?:{'|'.join(deb_bon_list)}) \b(?:{'|'.join(bon_list)}).{{0,40}}$",
        ]

    @Tag
    def PJ(self) -> str | list[str] | re.Pattern:
        """
        Tag associated with email attachment mentions.
        Ex: "See attached files"
        """
        return [
            r"\(See attached file: .{,60}?\..{1,4}\)",
            r"\(Embedded image(?: moved to file:)? .{,60}?\)",
            r"^.{,4}[\[\(][a-z0-9-_]*\.[a-zA-Z]{3,4}[\]\)]\s*$",
        ]

    @Tag
    def FOOTER(self) -> str | list[str] | re.Pattern:
        """
        Tag associated with email footer sentences.
        Ex: "Envoyé de mon iPhone"
        """
        prefix = r"^.{0,40}"
        suffix = r".*"

        text_list = [
            # Français
            r"[aà] [l]['’ ]attention de.{0,80}$",
            r"Les informations contenues dans ce courrier [ée]lectronique et toutes les pi[eè]ces qui y sont jointes",
            r"Le pr[ée]sent document est couvert par le secret professionnel",
            r"Ce message et toutes les pi[eè]ces jointes sont confidentiels",
            r"Ce message et les fichiers [ée]ventuels",
            r"Ce message est confidentiel",
            r"Ce message contient des informations confidentielles"
            r"Si vous avez recu ce message ([ée]lectronique )?par erreur",
            r"Toute modification, [ée]dition, utilisation",
            r"Tout usage, communication ou reproduction",
            r"Il ne peut etre lu, ni copi[ée], ni communiqu[ée], ni utilis[ée]",
            r"L'[ée]metteur d[ée]cline toute responsabilit[ée]",
            r"Sauf mention contraire, le present message",
            r"Afin de faciliter nos echanges et optimiser le traitement",
            r"Afin de contribuer au respect de l'environnement",
            r"N'imprimez ce message que si cela est indispensable",
            r"Pensez a l'environnement avant d'imprimer ce message",
            r"Droit a la d[ée]connexion",
            r"Ceci est un mail automatique",
            r"Les formats de fichiers acceptés sont : PDF, DOC, DOCX, JPEG, JPG, TIFF, TXT, ODT, XLS, XLSX",
            r"Tout autre format de fichiers ne sera pas transmis au dossier",
            # English
            r"This message and any attachments are confidential",
            r"This e-mail and any files transmitted",
            r"If you have received this (?:message|email) in error",
            r"Any unauthorized modification",
            r"The sender shall not be liable",
        ]

        diclaimer_regex_list = [f"{prefix}{x}{suffix}" for x in text_list]

        miscellaneous_footer_regex = [
            r"(?:courrier electronique|virus|antivirus){2,}",
            r"^.{0,10}Partag[ée] [aà].{0,5} partir de Word pour \b\w+\b$",
            r"^.{0,10}Provenance : Courrier pour Windows",
            r"^.{0,10}garanti sans virus.{0,30}",
            r"^.{0,10}www.avg.com",
            r"^.{0,10}www.avast.com",
            r"^.{0,10}T[ée]l[ée]charg.{0,10}$",
            r"^.{0,2}Obtenir{0,2}$",
            r"(?:Obtenez|T[ée]l[ée]charge[zr])? ?Outlook pour .*",
            r"^.{0,10}La visualisation des fichiers PDF n[ée]cessite.*",
            r"^.{0,10}Si vous recevez ce message par erreur",
            r"^.{0,10}Retrouvez-nous sur www\.maif-\w+\.fr",
            (
                r"^.{0,10}afin de contribuer au respect de l'environnement, merci de n'imprimer ce courriel qu'en c"
                r"as de n[ée]cessit[ée]"
            ),
            (
                r"^.{0,10}(?:Envoy[ée]|Numeris[ée]|Partag[ée]) de(?:puis)?\s*(?:mon)?\s*(?:mobile|smartphone|appareil|"
                r"\biP.|Galaxy|Yahoo|T[ée]l[ée]phone|(?:l'application))"
            ),
            r"^.{0,25}pour Android.{0,5}$",
            r"^.{0,5}Envoy[ée] avec.{0,10}$",
            r"^.{0,5}Envoy[ée] [àa] partir de.{0,35}$",
            r"^.{0,2}Courrier.{0,2}$",
            r"^.{0,2}Pour Windows.{0,2}$",
            r"^.{0,10}Scann[ée] avec.{,30}$",
            r"^.{,3}sans virus.{,3}$",
            # English
            r"^.{0,5}Sent with .*",
            r"^.{0,5}Sent from my .*",
        ]

        return diclaimer_regex_list + miscellaneous_footer_regex

    @Tag
    def THANKS(self) -> str | list[str] | re.Pattern:
        """
        Tag associated with email thanks sentences.
        Ex: "Merci beaucoup"
        """
        thanks_expressions = [
            r"(re)?(merci(e|ant)?(\s(d'|par)\s?avance)?)",
            r"thanks?( you)?",
            r"thx",
        ]
        thanks_pattern = r"\b(" + "|".join(thanks_expressions) + r")\b"

        exception_expressions = [
            r"de",
            r"d",
            r"mais",
            r"cependant",
            r"par contre",
            r"toutefois",
            r"pourtant",
            r"but",
            r"however",
        ]
        exception_pattern = r" *\b(" + "|".join(exception_expressions) + ") *"

        return [
            r"^.{0,20}" + thanks_pattern + r"(?!.{0,5}" + exception_pattern + r").{0,40}(?!.)",
        ]

    @Tag
    def SIGNATURE(self) -> str | list[str] | re.Pattern:
        """
        Tag associated with email signature sentences.
        Ex: "Tel : 0600000000"
        """

        # Jobs lines
        jobs = [
            r"associations?",
            r"(?:\w+ )?analyste?",
            r"conducteur",
            r"[ée]quipe",
            r"soci[ée]t[ée]",
            r"secr[ée]taires?",
            r"secr[ée]tariats?",
            r"directions?",
            r"services?",
            r"assistante?",
            r"gestionnaire",
            r"technicienn?e?",
            r"conseill[eè]re?",
            r"maitres?",
            r"avocats?",
            r"s\.a\.s\.",
            r"squad",
            r"charg[ée]e? d['e]\s*\w+(?:\s*\w+){,2}",
            # English
            r"Lead",
            r"Chief",
            r"VP",
            r"C.O",
            r"(Sales)? Representative",
        ]
        job_regex = r"\b(" + r"|".join(jobs) + r")\b"
        line_with_known_job = rf"(?:^ *.{{,5}}{self.word_block(1)}{job_regex}( +{self.word_block(6)})?(?:\n+|$))"

        # Street address regex
        street_word_list = [
            r"all[ée]e",
            r"avenue",
            r"boulevard",
            r"chemin",
            r"cours",
            r"[ée]splanade",
            r"h?ameau",
            r"impasse",
            r"lotissement",
            r"passage",
            r"place",
            r"square",
            r"quai",
            r"r[ée]sidence",
            r"rue",
            r"sentier",
            # English
            r"st\.?",
            r"street",
            r"ln\.?",
            r"lane",
            r"rd\.?",
            r"road",
            r"hill",
        ]
        street_word_pattern = r"\b(" + "|".join(street_word_list) + r")\b"

        # A number (house number) or range, free words (up to 2), an equivalent of street (rue, allée, etc)
        # and more free words (up to 5), free chars at the end (up to 2)
        street_address_regex = (
            r"^ *\d+(?:-\d+)?(?:bis|ter|b)?,? +(\w+\b *){,2}\b" + street_word_pattern + r"\b *(\w+\b[ -]*){,5}.{,2}$"
        )

        # Email address
        email_address_regex = r"(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"

        return [
            # Phone / Fax
            r"(?:^.{,3}(?:T[ée]l(?:[ée]phone)?\.?|mobile|phone|num[ée]ro|ligne).{,20}(?: *(?:\n+|$)))",
            (
                r"^(.{,10}:? ?\(?((?:\+|00)\(?33\)?(?: ?\(0\))?|0)\s*[1-"
                r"9]([\s.-]*\d{2}){4}.{,10}){,3}" + rf"({email_address_regex}.{{,10}})?"
                "( *(\n+|$))"
            ),
            r"^.{,3}(T[ée]l[ée]?(phone|copie)?|Fax|mobile|phone|num[ée]ro|ligne).{,20}$",
            r"^.{,3}Appel non surtax[ée].{,3}$",
            # Street / Address / Post code
            street_address_regex,
            r"^.{,3}Adresse.{,3}$",
            r"(?: *(?:BP|Boite Postale) *\d{,6} .{,30}(?: *(?:\n|$)))",
            r"(?: *\b\d{5}\b(?: *(?:\n|$))?(?: *(?:\S+(?: +\S+){,5})? *)(?: *(?:\n|$)))",
            # postal address with only street name and number :  EX : 9/11 rue Jeanne d'Arc
            r"^.{,3}\d+(?:[ /-]?\d+)? " + street_word_pattern + r".{,50}$",
            # EX number 2 : 23, rue de la Monnaie
            r"^.{,5}" + street_word_pattern + r".{,50}$",
            # postal address with only postal code and city name  :   EX : 76000 ROUEN
            r"^.{,3}\d{5}[\xa0| ][A-Z].{,5}$",
            # Known job title
            line_with_known_job,
            # Contact
            r"^.{,15}Pour nous contacter.{,15}$",
            r"^.{,3}Contact (e.mail|t[eé]l[eé]phone).{,3}$",
            # email adress EX: Adresse mail : cyrimmmman80@gmail.com
            r"^.{,3}([Aa]dresse mail|Mail).{,3}" + email_address_regex + r"$",
            # address with date EX : Torroella de Montgri, le 5 avril 2023
            r"^[A-Za-z]+(?: [A-Za-z]+)*, le \d{1,2} [A-Za-z]+ \d{4}.{,3}$",
        ]

    def post_process_tags(self, tags: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        Method to apply tagging rules posterior to the standard regex tagging.

        Parameters
        ----------
        tags: list[tuple[str, str]]
            Original tags

        Returns
        -------
        _: list[tuple[str, str]]
            Post-processed tags
        """
        # Signature lines containing first/last name
        tags = self.detect_name_signature(tags)

        return tags

    def detect_name_signature(self, tags: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        Method to detect lines containing First name / Surname
        Ex: Mr Joe Dupond

        Parameters
        ----------
        tags: list[tuple[str, str]]
            Original tags

        Returns
        -------
        _: list[tuple[str, str]]
            Post processed tags
        """
        # First name / Last name Signatures
        capitalized_words: str = r"[A-Z][-'A-za-zÀ-ÿ]{,10}"
        particles: str = r"le|d[ei]|d'?|v[oa]n|del"
        line_with_name: str = (
            rf"(?:^[ >]*{capitalized_words}(?:-{capitalized_words})?(?:(?: +(?:{particles}"
            rf"))?\.? {{,2}}{capitalized_words}(?:-{capitalized_words})?){{1,4}}\.? *(?:\n+|$))"
        )

        # Forbidden words (lowercase)
        forbidden_words: set[str] = {"urgent", "attention"}

        new_tags: list[tuple[str, str]] = list()
        for tag, text in tags:
            if tag == self.default_tag:
                match = re.match(line_with_name, text)
                has_forbidden_words: bool = bool(forbidden_words.intersection(text.lower().split()))

                if match and not has_forbidden_words:
                    tag = "SIGNATURE_NAME"

            new_tags.append((tag, text))

        return new_tags


class TransferredEmailProcessor(MelusineTransformer):
    """
    Processing specific to transferred emails such as:
    - Extracting the email address of the original sender (before transfer)
    - Removing empty messages related to the transfer action
    """

    def __init__(
        self,
        output_columns: Iterable[str] = ("messages", "det_clean_from"),
        tags_to_ignore: Iterable[str] = ("FOOTER", "SIGNATURE"),
        messages_column: str = "messages",
    ):
        """
        Parameters
        ----------
        output_columns: str
        tags_to_ignore: Iterable[str]
            If a message contains only tags in this list, it will be ignored
        messages_column: DataFrame column containing a list of Message instances
        """
        self.messages_column = messages_column
        input_columns = [self.messages_column]

        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.process_transfered_mail,
        )

        self.tags_to_ignore = tuple(tags_to_ignore)
        self.json_exclude_list.append("input_columns")

    @property
    def email_pattern(self) -> str:
        """
        Regex pattern to detect an email address.
        """
        return r"\w+(?:[-+.']\w+)*@\w+(?:[-.]\w+)*\.\w+(?:[-.]\w+)*"

    @property
    def meta_email_address_regex(self) -> str:
        """
        Regex to extract an email address from an email transition pattern.
        Ex:
            De: jane@gmail.fr A: joe@gmail.fr Envoyé à: 11h22
            => jane@gmail.fr
        """
        # De: jane@gmail.fr A: joe@gmail.fr Envoyé à: 11h22
        # De:\nJane <jane@gmail.fr>\nA: joe@gmail.fr Envoyé à: 11h22
        # De: Jane "jane@gmail.fr"\nA: joe@gmail.fr Envoyé à: 11h22
        start_pattern = r"(?:Message de\b|\bDe\b|Exp[ée]diteur|\bFrom\b) ?\n? ?: *\n?(?:.{,30}[ <\"])?"
        email_pattern = self.email_pattern
        end_pattern = r"(?: *<.*>)?(?:.{,5}(?:$|\n)|.{,5}(?:Envoy|A|Sent|À))"
        meta_pattern_1 = f"{start_pattern}({email_pattern}){end_pattern}"

        # Le 28 févr. 2023 à 10:33, joee@gmail.fr a écrit :
        # Le dim., 28 févr. 2023 à 10:33, joee@gmail.fr a écrit :
        # Le 01/01/2001 10:33, Joe <joee@gmail.fr> a écrit :
        start_pattern = r"Le (?:\d.{,15}[aà]|[a-z]{3}.{,50}[aà]|\d{2}/\d{2}/\d{4}) \d{,2}[:hH]\d{,2}.{,30} <?"
        end_pattern = r">? a [ée]crit ?:"
        meta_pattern_2 = f"{start_pattern}({email_pattern}){end_pattern}"

        # Assemble patterns
        meta_pattern = "|".join([meta_pattern_1, meta_pattern_2])

        return meta_pattern

    def process_transfered_mail(self, message_list: list[Message]) -> tuple[list[Message], str | None]:
        """
        Run all transformations related to transfer emails.

        Args:
            message_list: Emails input data

        Returns:
            message_list: List of messages in the conversation
            clean_address_from: Processed sender email address
        """
        clean_address_from: str | None = None

        # Filter out transfer message (contains only irrelevant tags)
        message_list = self.filter_message_list(message_list)

        # Extract email address data from transition pattern
        top_message = message_list[0]
        extracted_address_from: str | None = self.extract_email_address(top_message)

        # If no address
        if extracted_address_from:
            clean_address_from = extracted_address_from

        return message_list, clean_address_from

    def extract_email_address(self, message: Message) -> str | None:
        """
        Extract sender email address from message meta (transition pattern).

        Args:
            message: Message with text and metadata

        Returns:
            extracted_address_from: Extracted sender address if available
        """
        extracted_address_from = None

        if message.meta:
            # Extract email address
            match_list = re.findall(self.meta_email_address_regex, message.meta)

            # Filter out empty matches
            match_list = [match for match_group in match_list for match in match_group if match]

            # Sanity check on address
            if match_list and "@" in match_list[0]:
                extracted_address_from = match_list[0]

        return extracted_address_from

    def filter_message_list(self, message_list: list[Message]) -> list[Message]:
        """ """
        top_message = message_list[0]

        parts = top_message.extract_parts()
        contains_only_tags_to_ignore = all([tag.startswith(self.tags_to_ignore) for tag, _ in parts])

        if contains_only_tags_to_ignore and (len(message_list) > 1):
            message_list = message_list[1:]

        return message_list


class DeterministicTextFlagger(MelusineTransformer):
    """
    Class to flag text patterns such as :
    "new york" => "new_york"
    """

    def __init__(
        self,
        text_flags: dict[str, Any],
        input_columns: str = "text",
        output_columns: str = "text",
        remove_multiple_spaces: bool = True,
        add_spaces: bool = True,
    ):
        """
        Parameters
        ----------
        text_flags: dict[str, Any]
            Dict containing flag name as key and regex pattern as value
        add_spaces: bool
            If true, add spaces around flags
        remove_multiple_spaces: bool
            If True, remove multiple spaces after flagging
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.flag_text,
        )
        self.text_flags = text_flags
        self.add_spaces = add_spaces
        self.remove_multiple_spaces = remove_multiple_spaces

    @staticmethod
    def default_flag_text(
        text: str,
        flag_dict: dict[str, str],
        add_spaces: bool = True,
        remove_multiple_spaces: bool = True,
    ) -> str:
        """
        Method to apply flagging on a text.
        General flagging: replace remarkable expressions by a flag
        Ex: 0123456789 => flag_phone_

        Parameters
        ----------
        flag_dict: dict[str, str]
            Flagging dict with regex as key and replace_text as value
        text: str
            Text to be flagged
        add_spaces: bool
            If true, add spaces around flags
        remove_multiple_spaces: bool
            If True, remove multiple spaces after flagging

        Returns
        -------
        text: str
            Flagged text
        """
        # Support for nested flag dicts
        for key, value in flag_dict.items():
            if isinstance(value, dict):
                text = DeterministicTextFlagger.default_flag_text(
                    text=text,
                    flag_dict=value,
                    add_spaces=add_spaces,
                    remove_multiple_spaces=remove_multiple_spaces,
                )
            else:
                # Add spaces to avoid merging words with flags
                if add_spaces:
                    replace_value = " " + value + " "
                else:
                    replace_value = value
                text = re.sub(key, replace_value, text, flags=re.I)

            if remove_multiple_spaces:
                text = re.sub(" +", " ", text)
        return text.strip()

    def flag_text(self, text: str) -> str:
        """
         Method to flag text.

         Parameters
         ----------
         text: str
             Text to be flagged

         Returns
         -------
        _: str
             Flagged text
        """
        # Join collocations
        text = self.default_flag_text(
            text,
            self.text_flags,
            add_spaces=self.add_spaces,
            remove_multiple_spaces=self.remove_multiple_spaces,
        )

        return text


class Cleaner(MelusineTransformer):
    """
    Class to clean text columns
    """

    def __init__(
        self,
        substitutions: dict[str, Any],
        input_columns: str = "text",
        output_columns: str = "text",
    ):
        """
        Parameters
        ----------
        substitutions: dict[str, Any]
            Dict containing replace pattern and replacement value
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.clean,
        )
        self.substitutions = substitutions

    def clean(self, text: str) -> str:
        """
         Method to clean text.

         Parameters
         ----------
         text: str
             Text to be flagged

         Returns
         -------
        _: str
             Flagged text
        """
        # Join collocations
        text = DeterministicTextFlagger.default_flag_text(
            text,
            self.substitutions,
            add_spaces=False,
            remove_multiple_spaces=True,
        )

        return text


class DateProcessor(MelusineTransformer):

    """
    Parse string date to iso format string date
    """

    ISO_FORMAT = "%Y-%m-%d"
    LANGUAGE = ["en_US", "fr_FR", "es_ES", "it_IT", "nl_NL", "de_DE", "tr_TR"]
    CALENDAR_ABBR = {
        "lu,": "lun",
        "ma,": "mar",
        "me,": "mer",
        "je,": "jeu",
        "jeudii": "jeudi",
        "ve,": "ven",
        "sa,": "sam",
        r"^di,": "dim",
        r"\sjune\s": " jun ",
        r"\sjuly\s": " jul ",
    }
    FRENCH_CALENDAR_ABBR = {
        "fev ": "févr ",
        "jun": "juin ",
        "jul ": "juil ",
        "aout": "août",
        "sep ": "sept ",
        "dec ": "déc ",
        "iuillet": "juillet",
        "juihet": "juillet",
        "sptembre": "septembre",
        "seplentre": "septembre",
        "septembm": "septembre",
        "septernbre": "septembre",
        "juillat": "juillet",
        "decembre": "décembre",
        "fevrier": "février",
    }
    PATTERN_DATE = {
        r"(\d{4}-\d{2}-\d{2})": "YYYY-MM-DD",
        r"(\d{4}/\d{2}/\d{2})": "YYYY/MM/DD",
        r"(\d{2}-\d{2}-\d{4})": "DD-MM-YYYY",
        r"(\d{2}/\d{2}/\d{4})": "DD/MM/YYYY",
        r"([A-zÀ-ÿ]{4,10}, \d{1,2} [A-zÀ-ÿ]{3,4}, \d{4})": "dddd, DD MMM, YYYY",
        r"([A-zÀ-ÿ]{4,10}, \d{1,2} [A-zÀ-ÿ]{3,4} \d{4})": "dddd, DD MMM YYYY",
        r"([A-zÀ-ÿ]{4,10} \d{1,2} [A-zÀ-ÿ]{3,4} \d{4})": "dddd DD MMM YYYY",
        r"([A-zÀ-ÿ]{4,10}, \d{1,2} [A-zÀ-ÿ]{,10}, \d{4})": "dddd, DD MMMM, YYYY",
        r"([A-zÀ-ÿ]{4,10}, \d{1,2} [A-zÀ-ÿ]{,10} \d{4})": "dddd, DD MMMM YYYY",
        r"([A-zÀ-ÿ]{4,10} \d{1,2} [A-zÀ-ÿ]{,10} \d{4})": "dddd DD MMMM YYYY",
        r"(\d{1,2} [A-zÀ-ÿ]{3,4} \d{4})": "DD MMM YYYY",
        r"(\d{1,2} [A-zÀ-ÿ]{3,4}, \d{4})": "DD MMM, YYYY",
        r"(\d{1,2} [A-zÀ-ÿ]{,10} \d{4})": "DD MMMM YYYY",
        r"(\d{1,2} [A-zÀ-ÿ]{,10}, \d{4})": "DD MMMM, YYYY",
        r"([A-zÀ-ÿ]{3}, \d{1,2} [A-zÀ-ÿ]{3}, \d{4})": "ddd, DD MMM, YYYY",
        r"([A-zÀ-ÿ]{3}, \d{1,2} [A-zÀ-ÿ]{3} \d{4})": "ddd, DD MMM YYYY",
        r"([A-zÀ-ÿ]{3}\. \d{1,2} [A-zÀ-ÿ]{,10}\.? \d{4})": "ddd DD MMM YYYY",
        r"([A-zÀ-ÿ]{2}, \d{1,2} [A-zÀ-ÿ]{3}, \d{4})": "ddd, DD MMM, YYYY",
        r"([A-zÀ-ÿ]{2}, \d{1,2} [A-zÀ-ÿ]{3} \d{4})": "ddd, DD MMM YYYY",
        r"(\d{1,2} [A-zÀ-ÿ]{3} \d{4})": "DD MMM YYYY",
    }

    def __init__(
        self,
        input_columns: str = "date",
        output_columns: str = "date",
    ) -> None:
        """
        Parameters
        ----------
        input_columns: str
            Input columns for the transform operation
        output_columns: str
            Outputs columns for the transform operation
        """
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.parse_date_to_iso,
        )

    @classmethod
    def parse_date_to_iso(cls, date_: str) -> str | None:
        """
        This function use the package arrow to convert a date from string format with any
        type of format (i.e. vendredi 8 juillet 2020 -> 2020-07-08)
        This package is prefered to datetime because datetime use locale settings and
        arrow do not use locale setting
        Visit https://arrow.readthedocs.io/en/latest/ for more information

        Parameters
        ----------
        date_: str
                date read from mail with any format

        Returns
        -------
        date_: str
                date_ as string with iso format (YYYY-MM-DD)
        """
        # Initialization
        matched_group: str | None = None
        date_ = date_ or ""
        date_ = date_.lower()

        for pattern, format_ in cls.PATTERN_DATE.items():
            pattern_compiled = re.compile(pattern)
            matched_date_format = pattern_compiled.search(date_)
            if matched_date_format is not None:
                matched_group = matched_date_format.group()

                # Replace single digit by 0+digit (i.e. 9 -> 09)
                matched_group = cls.process_single_digit(matched_group, pattern)

                # Replace some known abbreviations
                for key, value in cls.CALENDAR_ABBR.items():
                    matched_group = re.sub(key, value, matched_group)

                matched_group = cls.convert_to_iso_format(matched_group, format_)

                # The format matched so we end the for loop
                break

        return matched_group

    @staticmethod
    def process_single_digit(matched_group: str, pattern: str) -> str:
        """
        Replace single digit by 0+digit
        i.e.: 9 -> 09
        """
        if r"\d{1,2}" in pattern:
            # Case when digit is in the middle of the string
            number_searched = re.search(r"\s\d\s", matched_group)
            if number_searched is not None:
                number = number_searched.group().strip()
                matched_group = matched_group.replace(f" {number} ", f" 0{number} ")
            # Case when digit is the first char of the string
            number_searched = re.search(r"^\d\s", matched_group)
            if number_searched is not None:
                number = number_searched.group().strip()
                matched_group = re.sub(r"^\d\s", f"0{number} ", matched_group)
        return matched_group

    @classmethod
    def convert_to_iso_format(cls, matched_group: str, format_: str) -> str | None:
        """
        Try to convert the date found as any string form to ISO format
        """
        # In case we are working with abbreviations
        if "ddd" in format_ or "MMM" in format_:
            for lang in cls.LANGUAGE:
                try:
                    matched_group_copy = matched_group
                    if lang == "fr_FR":
                        matched_group_copy = re.sub("[^A-zÀ-ÿ0-9:, ]+", "", matched_group_copy)
                        for key, value in cls.FRENCH_CALENDAR_ABBR.items():
                            matched_group_copy = re.sub(key, value, matched_group_copy)
                    matched_group = arrow.get(matched_group_copy, format_, locale=lang).datetime.strftime(
                        cls.ISO_FORMAT
                    )
                    break
                except arrow.parser.ParserMatchError:
                    pass

            # ISO Format
            if re.search(r"\d{4}-\d{2}-\d{2}", matched_group):
                return matched_group

            # We failed finding the abbreviation so we give up and return None
            return None

        # We were not working with abbrevations so we use arrow package
        return arrow.get(matched_group, format_).datetime.strftime(cls.ISO_FORMAT)
