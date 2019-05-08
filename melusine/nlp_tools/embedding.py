import logging
from gensim.models import Word2Vec
from melusine.utils.streamer import Streamer

log = logging.getLogger('Embeddings')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%d/%m %I:%M')


class Embedding():
    """Class to train embeddings with Word2Vec algorithm.

    Attributes
    ----------
    input_column : str,
        Input text column to consider for the embedding.

    stream : Streamer instance,
        Builds a stream a tokens from a pd.Dataframe to train the embeddings.

    embedding : Word2Vec instance from Gensim

    Examples
    --------
    >>> from melusine.nlp_tools.embedding import Embedding
    >>> embedding = Embedding()
    >>> embedding.train(X)
    >>> embedding.save(filepath)
    >>> embedding = Embedding().load(filepath)

    """

    def __init__(self,
                 input_column='clean_text',
                 workers=40,
                 seed=42,
                 iter=15,
                 size=300,
                 window=5,
                 min_count=100):
        self.logger = logging.getLogger('NLUtils.Embedding')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.debug('Create an Embedding instance.')
        self.input_column = input_column
        self.streamer = Streamer(column=self.input_column)
        self.workers = workers
        self.seed = seed
        self.iter = iter
        self.size = size
        self.window = window
        self.min_count = min_count

    def save(self, filepath):
        """Method to save Embedding object."""
        self.embedding.save(filepath)

    def load(self, filepath):
        """Method to load Embedding object."""
        self.embedding = Word2Vec.load(filepath)
        return self

    def train(self, X):
        """Train embeddings with Word2Vec algorithm.

        Parameters
        ----------
        X : pd.Dataframe
            Containing a clean body column.

        Returns
        -------
        self : object
            Returns the instance
        """
        self.logger.info('Start training for embedding')
        self.streamer.to_stream(X)
        self.embedding = Word2Vec(self.streamer.stream,
                                  workers=self.workers,
                                  seed=self.seed,
                                  iter=self.iter,
                                  size=self.size,
                                  window=self.window,
                                  min_count=self.min_count)
        self.logger.info('Done.')
        pass
