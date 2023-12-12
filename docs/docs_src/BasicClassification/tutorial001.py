from typing import List

# --8<-- [start:create_dataset]
import pandas as pd
from transformers import pipeline

from melusine.base import MelusineDetector


def create_dataset():
    df = pd.DataFrame(
        [
            {
                "header": "Dossier 123456",
                "body": "Merci beaucoup pour votre gentillesse et votre écoute !",
            },
            {
                "header": "Réclamation (Dossier 987654)",
                "body": ("Bonjour, je ne suis pas satisfait de cette situation, " "répondez-moi rapidement svp!"),
            },
        ]
    )

    return df


# --8<-- [end:create_dataset]


def transformers_standalone():
    # --8<-- [start:transformers]
    model_name_or_path = "cmarkea/distilcamembert-base-nli"

    sentences = [
        "Quelle belle journée aujourd'hui",
        "La marée est haute",
        "Ce film est une catastrophe, je suis en colère",
    ]

    classifier = pipeline(task="zero-shot-classification", model=model_name_or_path, tokenizer=model_name_or_path)

    result = classifier(
        sequences=sentences, candidate_labels=", ".join(["positif", "négatif"]), hypothesis_template="Ce texte est {}."
    )
    # --8<-- [end:transformers]

    return result


# --8<-- [start:detector_init]
class DissatisfactionDetector(MelusineDetector):
    """
    Detect if the text expresses dissatisfaction.
    """

    # Dataframe column names
    OUTPUT_RESULT_COLUMN = "dissatisfaction_result"
    TMP_DETECTION_INPUT_COLUMN = "detection_input"
    TMP_DETECTION_OUTPUT_COLUMN = "detection_output"

    # Model inference parameters
    POSITIVE_LABEL = "positif"
    NEGATIVE_LABEL = "négatif"
    HYPOTHESIS_TEMPLATE = "Ce texte est {}."

    def __init__(self, model_name_or_path: str, text_columns: List[str], threshold: float):
        self.text_columns = text_columns
        self.threshold = threshold
        self.classifier = pipeline(
            task="zero-shot-classification", model=model_name_or_path, tokenizer=model_name_or_path
        )

        super().__init__(input_columns=text_columns, output_columns=[self.OUTPUT_RESULT_COLUMN], name="dissatisfaction")
        # --8<-- [end:detector_init]

    # --8<-- [start:pre_detect]
    def pre_detect(self, row, debug_mode=False):
        # Assemble the text columns into a single text
        effective_text = ""
        for col in self.text_columns:
            effective_text += "\n" + row[col]
        row[self.TMP_DETECTION_INPUT_COLUMN] = effective_text

        # Store the effective detection text in the debug data
        if debug_mode:
            row[self.debug_dict_col] = {"detection_input": row[self.TMP_DETECTION_INPUT_COLUMN]}

        return row
        # --8<-- [end:pre_detect]

    # --8<-- [start:detect]
    def detect(self, row, debug_mode=False):
        # Run the classifier on the text
        pipeline_result = self.classifier(
            sequences=row[self.TMP_DETECTION_INPUT_COLUMN],
            candidate_labels=", ".join([self.POSITIVE_LABEL, self.NEGATIVE_LABEL]),
            hypothesis_template=self.HYPOTHESIS_TEMPLATE,
        )
        # Format classification result
        result_dict = dict(zip(pipeline_result["labels"], pipeline_result["scores"]))
        row[self.TMP_DETECTION_OUTPUT_COLUMN] = result_dict

        # Store ML results in the debug data
        if debug_mode:
            row[self.debug_dict_col].update(result_dict)

        return row
        # --8<-- [end:detect]

    # --8<-- [start:post_detect]
    def post_detect(self, row, debug_mode=False):
        # Compare classification score to the detection threshold
        if row[self.TMP_DETECTION_OUTPUT_COLUMN][self.NEGATIVE_LABEL] > self.threshold:
            row[self.OUTPUT_RESULT_COLUMN] = True
        else:
            row[self.OUTPUT_RESULT_COLUMN] = False

        return row

    # --8<-- [end:post_detect]


def run():
    # --8<-- [start:run]
    df = create_dataset()

    detector = DissatisfactionDetector(
        model_name_or_path="cmarkea/distilcamembert-base-nli",
        text_columns=["header", "body"],
        threshold=0.7,
    )

    df = detector.transform(df)
    # --8<-- [end:run]

    # Debug mode
    df = create_dataset()
    df.debug = True
    _ = detector.transform(df)

    return df


if __name__ == "__main__":  # pragma no cover
    run()
