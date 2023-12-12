from typing import Callable, List

import pandas as pd

from melusine.base import BaseMelusineDetector


# --8<-- [start:detector]
class MyCustomDetector(BaseMelusineDetector):
    @property
    def transform_methods(self) -> List[Callable]:
        return [self.prepare, self.run]

    def prepare(self, row, debug_mode=False):
        return row

    def run(self, row, debug_mode=False):
        row[self.output_columns[0]] = "12345"
        return row


# --8<-- [end:detector]


def run():
    # --8<-- [start:run]
    df = pd.DataFrame(
        [
            {"input_col": "test1"},
            {"input_col": "test2"},
        ]
    )

    detector = MyCustomDetector(input_columns=["input_col"], output_columns=["output_col"], name="custom")

    df = detector.transform(df)
    # --8<-- [end:run]

    return df
