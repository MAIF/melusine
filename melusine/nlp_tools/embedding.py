import logging
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from melusine.core.melusine_transformer import MelusineTransformer


logger = logging.getLogger(__name__)


class Embedding(MelusineTransformer):
    """ """

    FILENAME = "gensim_embeddings_meta.json"
    KEYED_VECTORS_FILENAME = "gensim_embeddings.w2v"

    def __init__(
        self, input_columns=("tokens",), output_columns=("tokens",), **embeddings_args
    ):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=None,
        )
        self.embeddings_args = embeddings_args
        self.embeddings_ = None
        self.EXCLUDE_LIST.append("embeddings_")

    def fit(self, df, y=None):
        """ """
        input_data = df[self.input_columns[0]]
        w2v = Word2Vec(**self.embeddings_args)
        w2v.build_vocab(input_data)

        w2v.train(
            input_data,
            total_examples=w2v.corpus_count,
            epochs=w2v.epochs,
        )

        self.embeddings_ = w2v.wv

        return self

    def transform(self, df):
        return df

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the Embeddings into a pkl file

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            Prefix for saved files.
        """
        d = self.__dict__.copy()

        self.save_json(
            save_dict=d,
            path=path,
            filename_prefix=filename_prefix,
        )

        filepath = self.get_file_path(
            self.KEYED_VECTORS_FILENAME, path, filename_prefix
        )
        self.embeddings_.save(filepath)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load the Embeddings from a json file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str
        Returns
        -------
        _: Phraser
            Phraser instance
        """
        # Load json file
        config_dict = cls.load_json(path, filename_prefix=filename_prefix)
        instance = cls(**config_dict)

        # Load embeddings file
        filepath = cls.search_file(
            cls.KEYED_VECTORS_FILENAME, path, filename_prefix=filename_prefix
        )
        embeddings = KeyedVectors.load(filepath)
        instance.embeddings_ = embeddings

        return instance
