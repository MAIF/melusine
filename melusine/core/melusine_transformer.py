from sklearn.base import BaseEstimator, TransformerMixin

from melusine.core.base_melusine_class import BaseMelusineClass
from melusine.backend.melusine_backend import backend


class MelusineTransformer(BaseMelusineClass, BaseEstimator, TransformerMixin):
    """
    Base transformer class.
    Melusine transformers have the following characteristics:
    - Can be added to a Melusine (or sklearn) pipeline
    - Can be saved and loaded
    """

    def __init__(self, input_columns, output_columns, func=None):
        super().__init__()
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.func = func

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if not self.func:
            return AttributeError(
                f"Instance of {type(self)} does not have a func attribute\n"
                "You should either specify a func attribute or define your own transform method"
            )
        return backend.apply_transform(
            data=df,
            input_columns=self.input_columns,
            output_columns=self.output_columns,
            func=self.func,
        )
