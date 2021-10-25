import logging

from melusine.core.melusine_transformer import MelusineTransformer

logger = logging.getLogger(__name__)


class DummyLemmatizer(MelusineTransformer):
    def __init__(self, input_columns="tokens", output_columns="tokens"):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.lemmatize,
        )

    @staticmethod
    def lemmatize_token(token):
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        else:
            return token

    def lemmatize(self, tokens):
        # Join collocations
        tokens = [self.lemmatize_token(token) for token in tokens]

        return tokens

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the Token Flagger into a pkl file

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            Prefix for saved files.
        """
        self.save_pkl(path, filename_prefix)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load the DeterministicTextFlagger from a json file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str
        Returns
        -------
        _: DummyLemmatizer
            DummyLemmatizer instance
        """
        return cls.load_pkl(path, filename_prefix=filename_prefix)
