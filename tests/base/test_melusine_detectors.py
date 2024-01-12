from typing import Callable, List

import pandas as pd

from melusine.base import BaseMelusineDetector


class MyDetector(BaseMelusineDetector):
    @property
    def transform_methods(self) -> List[Callable]:
        return [self.row_method, self.df_method]

    def row_method(self, row, debug_mode=False):
        input_data = row[self.input_columns[0]]
        row[self.output_columns[0]] = input_data + "_row"
        return row

    def df_method(self, df, debug_mode=False):
        df[self.output_columns[1]] = df[self.input_columns[0]].str.upper() + "_df"
        return df


def test_detector_transform_dataframe_wise():
    df = pd.DataFrame([{"input_col": "test0"}, {"input_col": "test1"}])
    detector = MyDetector(name="test_detector", input_columns=["input_col"], output_columns=["row_output", "df_output"])
    df = detector.transform(df)

    assert df["row_output"].iloc[0] == "test0_row"
    assert df["row_output"].iloc[1] == "test1_row"
    assert df["df_output"].iloc[0] == "TEST0_df"
    assert df["df_output"].iloc[1] == "TEST1_df"
