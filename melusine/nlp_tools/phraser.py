import logging
import gensim
from melusine.utils.io_utils import get_file_path, save_pkl_generic, load_pkl_generic

logger = logging.getLogger(__name__)


class Phraser:
    """
    Train and use a Gensim Phraser.
    """

    FILENAME = "gensim_phraser_meta.pkl"
    PHRASER_FILENAME = "gensim_phraser"

    def __init__(self, input_columns="tokens", output_columns="tokens", **phraser_args):
        if isinstance(input_columns, str):
            self.input_columns = (input_columns,)
        else:
            self.input_columns = input_columns

        if isinstance(output_columns, str):
            self.output_columns = (output_columns,)
        else:
            self.output_columns = output_columns

        self.phraser_args = phraser_args
        self.phraser_ = None

    def fit(self, df, y=None):
        """ """
        input_data = df[self.input_columns[0]]
        phrases = gensim.models.Phrases(input_data, **self.phraser_args)
        self.phraser_ = gensim.models.phrases.Phraser(phrases)

        return self

    def transform(self, df):
        df[self.input_columns[0]] = self.phraser_[df[self.output_columns[0]]]
        return df

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
        filepath = get_file_path(self.PHRASER_FILENAME, path, filename_prefix)
        self.phraser_.save(filepath)

        self.phraser_ = None
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
        _: Phraser
            Phraser instance
        """
        # Load Phraser class
        instance = load_pkl_generic(cls.FILENAME, path, filename_prefix=filename_prefix)

        # Load Gensim Phraser object
        filepath = get_file_path(cls.PHRASER_FILENAME, path, filename_prefix)
        phraser = gensim.models.phrases.Phraser.load(filepath)
        instance.phraser_ = phraser

        return instance
