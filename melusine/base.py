"""
Base classes of the Melusine framework.

Implemented classes: [
    MelusineTransformer,
    MelusineDetector,
    MelusineModel,
    BaseLabelProcessor,
    MissingModelInputFieldError,
    MissingFieldError,
    MelusineFeatureEncoder
]
"""
from __future__ import annotations

import copy
import inspect
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, TypeVar, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from melusine.backend import backend
from melusine.io import IoMixin

logger = logging.getLogger(__name__)

# Dataset types supported by Melusine : pandas DataFrame and dicts
MelusineDataset = Union[Dict[str, Any], pd.DataFrame]

# Corresponding items are:
# - Dataset : Pandas DataFrame => Item : Pandas Series
# - Dataset Dict => Item Dict
MelusineItem = Union[Dict[str, Any], pd.Series]
Transformer = TypeVar("Transformer", bound="MelusineTransformer")


class TransformError(Exception):
    """
    Exception raised when an error occurs during the transform operation.
    """


class MelusineTransformer(BaseEstimator, TransformerMixin, IoMixin):
    """
    Define a MelusineTransformer object.

    Is an abstract class.

    It can be a Processor or a Detector.
    """

    def __init__(
        self,
        input_columns: str | Iterable[str],
        output_columns: str | Iterable[str],
        func: Callable | None = None,
    ) -> None:
        """
        Attribute initialization.

        Parameters
        ----------
        input_columns: Union[str, Iterable[str]]
            List of input columns
        output_columns: Union[str, Iterable[str]]
            List of output columns
        func: Callable
            Transform function to be applied
        """
        IoMixin.__init__(self)

        self.input_columns: list[str] = self.parse_column_list(input_columns)
        self.output_columns: list[str] = self.parse_column_list(output_columns)
        self.func = func

    @staticmethod
    def parse_column_list(columns: str | Iterable[str]) -> list[str]:
        """
        Transform a string into a list with a single element.

        Parameters
        ----------
        columns: Union[str, Iterable[str]]
            String or list of strings with column name(s).

        Returns
        -------
        _: list[str]
            A list of column names.
        """
        # Change string into list of strings if necessary
        # "body" => ["body]
        if isinstance(columns, str):
            columns = [columns]
        return list(columns)

    def transform(self, data: MelusineDataset) -> MelusineDataset:
        """
        Transform input data.

        Parameters
        ----------
        data: MelusineDataset
            Input data.

        Returns
        -------
        _: MelusineDataset
            Transformed data (output).
        """
        logger.debug(f"Running transform for {type(self).__name__}")
        if self.func is None:
            raise AttributeError(f"Attribute func of MelusineTransformer {type(self).__name__} should not be None")
        try:
            return backend.apply_transform(
                data=data, input_columns=self.input_columns, output_columns=self.output_columns, func=self.func
            )

        except Exception as exception:
            func_name = self.func.__name__
            class_name = type(self).__name__
            input_columns = self.input_columns
            raise TransformError(
                f"Error in class: '{class_name}' "
                f"with method '{func_name}' "
                f"input_columns: {input_columns}\n"
                f"{str(exception)}"
            ).with_traceback(exception.__traceback__) from exception


class BaseMelusineDetector(MelusineTransformer, ABC):
    """
    Used to define detectors.

    Template Method str based on the MelusineTransformer class.
    """

    def __init__(
        self,
        name: str,
        input_columns: list[str],
        output_columns: list[str],
    ):
        """
        Attributes initialization.

        Parameters
        ----------
        name: str
            Name of the detector.
        input_columns:
            Detector input columns.
        output_columns:
            Detector output columns.
        """
        #  self.name needs to be set before the super class init
        #  Name is used to build the output_columns
        self.name = name

        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
        )

    @property
    def debug_dict_col(self) -> str:
        """
        Standard name for the column containing the debug info.

        Typically, a detector may return the following outputs:
        - output_result_col: bool
          > Ex: thanks_result: True
        - output_value_col: Any
          > Ex: thanks_output: "Remerciement plat"
        - output_score_col: float
          > Ex: thanks_score: 0.95
        - (debug) debug_dict_col: Dict[str, Any]
          > Ex: debug_thanks: {"thanks_text": "Merci"}
        """
        return f"debug_{self.name}"

    @property
    @abstractmethod
    def transform_methods(self) -> list[Callable]:
        """
        Specify the sequence of methods to be called by the transform method.

        Returns
        -------
        _: list[Callable]
            List of  methods to be called by the transform method.
        """

    def transform(self, df: MelusineDataset) -> MelusineDataset:
        """
        Re-definition of super().transform() => specific detector's implementation

        Transform input data.

        Parameters
        ----------
        df: MelusineDataset
            Input data.

        Returns
        -------
        _: MelusineDataset
            Transformed data (output).
        """
        logger.debug(f"Running transform for {type(self).__name__}")

        # Debug mode ON?
        debug_mode: bool = backend.check_debug_flag(df)

        # Validate fields of the input data
        self.validate_input_fields(df)

        # Work on a copy of the DataFrame and limit fields to effective input columns
        # data_ = backend.copy(data, fields=self.input_columns)

        # Work on a copy of the DataFrame and keep all columns
        # (too complex to handle model input columns)
        data_ = backend.copy(df)

        # Get list of new columns created by the detector
        return_cols = copy.deepcopy(self.output_columns)

        # Create debug data dict
        if debug_mode:
            data_ = backend.setup_debug_dict(data_, dict_name=self.debug_dict_col)
            return_cols.append(self.debug_dict_col)

        for method in self.transform_methods:
            logger.debug(f"Running transform for {type(self).__name__} ({method.__name__})")
            first_arg_name: str = list(inspect.signature(method).parameters)[0]

            if first_arg_name == "row":
                # Run row-wise method
                data_ = backend.apply_transform(
                    data=data_, input_columns=None, output_columns=None, func=method, debug_mode=debug_mode
                )
            else:
                data_ = method(data_, debug_mode=debug_mode)

        # Add new fields to the original MelusineDataset
        data = backend.add_fields(left=df, right=data_, fields=return_cols)

        return data

    def validate_input_fields(self, data: MelusineDataset) -> None:
        """
        Make sure that all the required input fields are present.

        Parameters
        ----------
        data: MelusineDataset
            Input data.
        """
        input_fields: list[str] = backend.get_fields(data)
        missing_fields: list[str] = [x for x in self.input_columns if x not in input_fields]
        if missing_fields:
            raise MissingFieldError(f"Fields {missing_fields} are missing from the input data")


class MelusineDetector(BaseMelusineDetector, ABC):
    """
    Defines an interface for detectors.
    All detectors used in a MelusinePipeline should inherit from the MelusineDetector class and
    implement the abstract methods.
    This ensures homogeneous coding style throughout the application.
    Alternatively, melusine user's can define their own Interface (inheriting from the BaseMelusineDetector)
    to suit their needs.
    """

    @property
    def transform_methods(self) -> list[Callable]:
        """
        Specify the sequence of methods to be called by the transform method.

        Returns
        -------
        _: list[Callable]
            List of  methods to be called by the transform method.
        """
        return [self.pre_detect, self.detect, self.post_detect]

    @abstractmethod
    def pre_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """What needs to be done before detection."""

    @abstractmethod
    def detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """Run detection."""

    @abstractmethod
    def post_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """What needs to be done after detection (e.g., mapping columns)."""


class MissingFieldError(Exception):
    """
    Exception raised when a missing field is encountered by a MelusineTransformer
    """


class MelusineRegex(ABC):
    """
    Class to standardise text pattern detection using regex.
    """

    REGEX_FLAGS: re.RegexFlag = re.IGNORECASE | re.MULTILINE

    # Match fields
    MATCH_RESULT: str = "match_result"
    NEUTRAL_MATCH_FIELD: str = "neutral_match_data"
    POSITIVE_MATCH_FIELD: str = "positive_match_data"
    NEGATIVE_MATCH_FIELD: str = "negative_match_data"

    # Match data
    MATCH_START: str = "start"
    MATCH_STOP: str = "stop"
    MATCH_TEXT: str = "match_text"

    def __init__(self, substitution_pattern: str = " ", default_match_group: str = "DEFAULT"):
        if not isinstance(substitution_pattern, str) or (len(substitution_pattern) > 1):
            raise ValueError(
                f"Parameter substitution_pattern should be a string of length 1, not {substitution_pattern}"
            )
        self.substitution_pattern = substitution_pattern
        self.default_match_group = default_match_group

    @property
    def regex_name(self) -> str:
        """
        Name of the Melusine regex object.
        Defaults to the class name.
        """
        return getattr(self, "_regex_name", type(self).__name__)

    @property
    @abstractmethod
    def positive(self) -> dict[str, str] | str:
        """
        Define regex patterns required to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """

    @property
    def neutral(self) -> dict[str, str] | str | None:
        """
        Define regex patterns to be ignored when running detection.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    def negative(self) -> dict[str, str] | str | None:
        """
        Define regex patterns prohibited to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    @abstractmethod
    def match_list(self) -> list[str]:
        """
        List of texts that should activate the MelusineRegex.

        Returns:
            _: List of texts.
        """

    @property
    @abstractmethod
    def no_match_list(self) -> list[str]:
        """
        List of texts that should NOT activate the MelusineRegex.

        Returns:
            _: List of texts.
        """

    def _get_match(
        self, text: str, base_regex: str | dict[str, str], regex_group: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Run specified regex on the input text and return a dict with matching group as key.

        Args:
            text: Text to apply regex on.
            base_regex: Regex to apply on text.
            regex_group: Name of the group the regex belongs to.

        Returns:
            Dict of regex matches for each regex group.
        """
        match_data_dict = {}

        if regex_group is None:
            regex_group = self.default_match_group

        if isinstance(base_regex, dict):
            for group, regex in base_regex.items():
                group_match_data = self._get_match(text, regex, group)
                match_data_dict.update(group_match_data)
        else:
            for match in re.finditer(base_regex, text, flags=self.REGEX_FLAGS):
                if not match_data_dict.get(regex_group):
                    match_data_dict[regex_group] = []

                # Get match position
                start, stop = match.span()

                match_data_dict[regex_group].append(
                    {
                        self.MATCH_START: start,
                        self.MATCH_STOP: stop,
                        self.MATCH_TEXT: text[start:stop],
                    }
                )

        return match_data_dict

    def ignore_text(
        self,
        text: str,
        match_data_dict: dict[str, list[dict[str, Any]]],
    ) -> str:
        """
        Replace neutral regex match text with substitution text to ignore it.

        Args:
            text: Input text.
            match_data_dict: Regex match results.

        Returns:
            _: Text with substituions.
        """
        for _, match_list in match_data_dict.items():
            for match_data in match_list:
                start = match_data[self.MATCH_START]
                stop = match_data[self.MATCH_STOP]

                # Mask text to ignore
                text = text[:start] + self.substitution_pattern * (stop - start) + text[stop:]

        return text

    def get_match_result(self, text: str) -> bool:
        """
        Apply MelusineRegex patterns (neutral, negative and positive) on the input text.
        Return a boolean output of the match result.

        Args:
            text: input text.

        Returns:
            _: True if the MelusineRegex matches the input text.
        """
        result = self(text)
        return result[self.MATCH_RESULT]

    def __call__(self, text: str) -> dict[str, Any]:
        """
        Apply MelusineRegex patterns (neutral, negative and positive) on the input text.
        Return a detailed output of the match results as a dict.

        Args:
            text: input text.

        Returns:
            _: Regex match results.
        """
        match_dict = {
            self.MATCH_RESULT: False,
            self.NEUTRAL_MATCH_FIELD: {},
            self.NEGATIVE_MATCH_FIELD: {},
            self.POSITIVE_MATCH_FIELD: {},
        }

        negative_match = False

        if self.neutral:
            neutral_match_data = self._get_match(text=text, base_regex=self.neutral)
            match_dict[self.NEUTRAL_MATCH_FIELD] = neutral_match_data

            text = self.ignore_text(text, neutral_match_data)

        if self.negative:
            negative_match_data = self._get_match(text=text, base_regex=self.negative)
            negative_match = bool(negative_match_data)
            match_dict[self.NEGATIVE_MATCH_FIELD] = negative_match_data

        positive_match_data = self._get_match(text=text, base_regex=self.positive)
        positive_match = bool(positive_match_data)
        match_dict[self.POSITIVE_MATCH_FIELD] = positive_match_data

        match_dict[self.MATCH_RESULT] = positive_match and not negative_match

        return match_dict

    def describe(self, text: str, position: bool = False) -> None:
        """
        User-friendly description of the regex match results.

        Args:
            text: Input text.
            position: If True, print regex match start and stop positions.
        """

        def _describe_match_field(match_field_data: dict[str, list[dict[str, Any]]]) -> None:
            """
            Format and print result description text.

            Args:
                match_field_data: Regex match result for a given field.
            """
            for group, match_list in match_field_data.items():
                for match_dict in match_list:
                    print(f"{indent}({group}) {match_dict[self.MATCH_TEXT]}")
                    if position:
                        print(f"{indent}start: {match_dict[self.MATCH_START]}")
                        print(f"{indent}stop: {match_dict[self.MATCH_STOP]}")

        indent = " " * 4
        match_data = self(text)

        if match_data[self.MATCH_RESULT]:
            print("The MelusineRegex match result is : POSITIVE")
        else:
            print("The MelusineRegex match result is : NEGATIVE")

        if not any(
            [
                match_data[self.NEUTRAL_MATCH_FIELD],
                match_data[self.NEGATIVE_MATCH_FIELD],
                match_data[self.POSITIVE_MATCH_FIELD],
            ]
        ):
            print("The input text did not match anything.")

        if match_data[self.NEUTRAL_MATCH_FIELD]:
            print("The following text was ignored:")
            _describe_match_field(match_data[self.NEUTRAL_MATCH_FIELD])

        if match_data[self.NEGATIVE_MATCH_FIELD]:
            print("The following text matched negatively:")
            _describe_match_field(match_data[self.NEGATIVE_MATCH_FIELD])

        if match_data[self.POSITIVE_MATCH_FIELD]:
            print("The following text matched positively:")
            _describe_match_field(match_data[self.POSITIVE_MATCH_FIELD])

    def test(self) -> None:
        """
        Test the MelusineRegex on the match_list and no_match_list.
        """
        for text in self.match_list:
            match = self(text)
            assert match[self.MATCH_RESULT] is True, f"Expected match for text\n{text}\nObtained: {match}"

        for text in self.no_match_list:
            match = self(text)
            assert match[self.MATCH_RESULT] is False, f"Expected no match for text:\n{text}\nObtained: {match}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(positive:{self.positive},neutral:{self.neutral},negative:{self.negative})"
