import logging

from melusine.utils.io_utils import save_pkl_generic, load_pkl_generic

logger = logging.getLogger(__name__)


class DummyLemmatizer:
    FILENAME = "lemmatizer.pkl"

    def __init__(
        self,
        input_columns=("tokens",),
        output_columns=("tokens",),
    ):
        if isinstance(input_columns, str):
            self.input_columns = (input_columns,)
        else:
            self.input_columns = input_columns

        if isinstance(output_columns, str):
            self.output_columns = (output_columns,)
        else:
            self.output_columns = output_columns

    @staticmethod
    def lemmatize_token(token):
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        else:
            return token

    def lemmatize(self, tokens):
        tokens = [self.lemmatize_token(token) for token in tokens]

        return tokens

    def fit(self, data, y=None):
        return self

    def transform(self, data):
        input_col = self.input_columns[0]
        output_col = self.output_columns[0]

        data[output_col] = data[input_col].apply(self.lemmatize)
        return data

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save object instance to a pickle file.

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            Prefix of the saved object
        """
        save_pkl_generic(self, self.FILENAME, path, filename_prefix)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load object instance from a pickle file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str
            Prefix of the saved object

        Returns
        -------
        _: DummyLemmatizer
            DummyLemmatizer instance
        """
        return load_pkl_generic(cls.FILENAME, path, filename_prefix=filename_prefix)
