import pandas as pd

from melusine.base import MelusineDetector


# --8<-- [start:detector]
# --8<-- [start:detector_init]
class MyVirusDetector(MelusineDetector):
    """
    Detect if the text expresses dissatisfaction.
    """

    # Dataframe column names
    OUTPUT_RESULT_COLUMN = "virus_result"
    TMP_DETECTION_INPUT_COLUMN = "detection_input"
    TMP_POSITIVE_REGEX_MATCH = "positive_regex_match"
    TMP_NEGATIVE_REGEX_MATCH = "negative_regex_match"

    def __init__(self, body_column: str, header_column: str):
        self.body_column = body_column
        self.header_column = header_column

        super().__init__(
            input_columns=[self.body_column, self.header_column],
            output_columns=[self.OUTPUT_RESULT_COLUMN],
            name="virus",
        )
        # --8<-- [end:detector_init]

    # --8<-- [start:pre_detect]
    def pre_detect(self, df, debug_mode=False):
        # Assemble the text columns into a single column
        df[self.TMP_DETECTION_INPUT_COLUMN] = df[self.header_column] + "\n" + df[self.body_column]

        return df
        # --8<-- [end:pre_detect]

    # --8<-- [start:detect]
    def detect(self, df, debug_mode=False):
        text_column = df[self.TMP_DETECTION_INPUT_COLUMN]
        positive_regex = r"(virus)"
        negative_regex = r"(corona[ _]virus)"

        # Pandas str.extract method on columns
        df[self.TMP_POSITIVE_REGEX_MATCH] = text_column.str.extract(positive_regex).apply(pd.notna)
        df[self.TMP_NEGATIVE_REGEX_MATCH] = text_column.str.extract(negative_regex).apply(pd.notna)

        return df
        # --8<-- [end:detect]

    # --8<-- [start:post_detect]
    def post_detect(self, df, debug_mode=False):
        # Boolean operation on pandas column
        df[self.OUTPUT_RESULT_COLUMN] = df[self.TMP_POSITIVE_REGEX_MATCH] & ~df[self.TMP_NEGATIVE_REGEX_MATCH]
        return df

    # --8<-- [end:post_detect]


# --8<-- [end:detector]


def run():
    # --8<-- [start:run]
    df = pd.DataFrame(
        [
            {"body": "This is a dangerous virus", "header": "test"},
            {"body": "test", "header": "test"},
            {"body": "test", "header": "viruses are dangerous"},
            {"body": "corona virus is annoying", "header": "test"},
        ]
    )

    detector = MyVirusDetector(body_column="body", header_column="header")

    df = detector.transform(df)
    # --8<-- [end:run]

    return df
