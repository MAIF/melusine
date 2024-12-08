"""
Classes of detectors.

Implemented classes: [ThanksDetector, VacationReplyDetector, ExpeditorDetector,
ReplyDetector, TransferDetector, RecipientsDetector]

"""

from typing import Any, Dict, List, Optional

from hugging_face.models.model import TextClassifier
from melusine.base import MelusineItem, MelusineRegex, MelusineTransformerDetector
from melusine.regex import DissatisfactionRegex


class DissatisfactionDetector(MelusineTransformerDetector):
    """
    Class to detect emails containing dissatisfaction emotion.

    Ex:
    je vous deteste,
    Cordialement
    """

    # Intermediate columns
    CONST_TEXT_COL_NAME: str = "effective_text"
    DISSATISFACTION_TEXT_COL: str = "dissatisfaction_text"
    CONST_DEBUG_TEXT_KEY: str = "text"
    CONST_DEBUG_PARTS_KEY: str = "parts"

    # Results columns
    DISSATISFACTION_ML_SCORE_COL: str = "dissatisfaction_ml_score"
    DISSATISFACTION_ML_MATCH_COL: str = "dissatisfaction_ml_result"
    DISSATISFACTION_BY_REGEX_MATCH_COL: str = "dissatisfaction_regex_result"

    def __init__(
        self,
        text_column: str,
        name: str,
        tokenizer_name_or_path: str,
        model_name_or_path: str,
        token: Optional[str] = None,
    ) -> None:
        """
        Attributes initialization.

        Parameters
        ----------
        text_column: str
            Name of the column containing the email text.
        name: str
            Name of the detector.
        tokenizer_name_or_path: str
            Name of model or path of the tokenizer.
        model_name_or_path: str
            Name of path of the model.
        text_column: str
            Name of the column containing the email text.
        token: Optional[str]
            hugging-face token .
        """

        # Input columns
        self.text_column = text_column
        input_columns: List[str] = [text_column]

        # Output columns
        self.result_column = f"{name}_result"
        output_columns: List[str] = [self.result_column]

        # Detection regex
        self.dissatisfaction_regex: MelusineRegex = DissatisfactionRegex()
        self.token = token

        super().__init__(
            name=name,
            input_columns=input_columns,
            output_columns=output_columns,
        )
        self.melusine_model = TextClassifier(
            tokenizer_name_or_path=tokenizer_name_or_path, model_name_or_path=model_name_or_path, token=self.token
        )

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

        # Last message body
        message_text: str = row[self.text_column]

        row[self.CONST_TEXT_COL_NAME] = "\n".join([message_text])

        # Prepare and save debug data
        if debug_mode:
            debug_dict: Dict[str, Any] = {
                self.CONST_DEBUG_TEXT_KEY: row[self.CONST_TEXT_COL_NAME],
            }
            row[self.debug_dict_col] = debug_dict

        return row

    def by_regex_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Use regex to detect dissatisfaction.

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
        detection_data = self.dissatisfaction_regex(text)
        detection_result = detection_data[self.dissatisfaction_regex.MATCH_RESULT]

        # Save debug data
        if debug_mode:
            debug_info[self.dissatisfaction_regex.regex_name] = detection_data
            row[self.debug_dict_col].update(debug_info)

        # Create new columns
        row[self.DISSATISFACTION_BY_REGEX_MATCH_COL] = detection_result
        return row

    def by_ml_detect(self, row: MelusineItem, debug_mode: bool = False) -> MelusineItem:
        """
        Use machine learning model to detect dissatisfaction.

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

        predictions, scores = self.melusine_model.predict(row[self.CONST_TEXT_COL_NAME])
        debug_info: Dict[str, Any] = {}

        row[self.DISSATISFACTION_ML_MATCH_COL], row[self.DISSATISFACTION_ML_SCORE_COL] = bool(predictions[0]), scores[0]
        # Save debug data
        if debug_mode:
            debug_info[self.DISSATISFACTION_ML_MATCH_COL] = row[self.DISSATISFACTION_ML_MATCH_COL]
            debug_info[self.DISSATISFACTION_ML_SCORE_COL] = row[self.DISSATISFACTION_ML_SCORE_COL]
            row[self.debug_dict_col].update(debug_info)
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
        ml_result = (row[self.DISSATISFACTION_ML_SCORE_COL] > 0.9) and row[self.DISSATISFACTION_ML_MATCH_COL]
        deterministic_result = row[self.DISSATISFACTION_BY_REGEX_MATCH_COL]
        row[self.result_column] = deterministic_result or ml_result
        return row
