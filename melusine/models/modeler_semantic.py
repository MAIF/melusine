import logging
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool

from sklearn.base import BaseEstimator, TransformerMixin
from melusine.utils.multiprocessing import apply_by_multiprocessing
from gensim.models import Word2Vec

def aggregation_percentile_60(x):
    """
    Explicit definition of the percentile 60 function.
    This is done because lambda function can not be used in multiprocessing (can't be pickled).
    NOT OK : aggregation_function_email_wise = lambda x: np.percentile(x, 60)
    OK : aggregation_function_email_wise = aggregation_percentile_60

    Parameters
    ----------
    x: list or numpy array
        list of scores associated with the tokens in the document
    """
    return np.percentile(x, 60)

class SemanticDetector(BaseEstimator, TransformerMixin):
    """Class to fit a Lexicon on an embedding and predicts the
    Attributes
    ----------
    base_seed_words : list,
        Seedwords list containing the seedwords for computing the Lexicons given by the User.
    seed_list : list,
        Same as base_seed_words but only the seeds present in the embedding vocabulary are kept.
    n_jobs : int,
        Number of CPUs to use to rate the emails.
    progress_bar : bool,
        Whether to print or not the progress bar while rating the emails.
    extend_seed_word_list : bool,
        Whether to use the seedwords as prefixes.
    seed_dict : dict,
        Filled if extend_seed_word_list==True. Key : prefix, Value : list of seedwords having this prefix in the vocabulary
    tokens_column : str,
        Name of the column in the Pandas Dataframe on which the polarity scores will be computed. Must be a column of tokens.
    normalize_scores : bool,
        Whether or not to normalize the lexicons' scores (so that they are centered around 0 and with a variance of 1)
    lexicon_dict : dict,
        Key : Seedword, Value : dict having all the embedding vocabulary as keys, and their semantic polarity value (cosine similarity) towards the seedword as values
    normalized_lexicon_dict : dict,
        Filled only if nomalize_scores==True. Contains the same as lexicon_dict, but with the normalized polarity values instead.
    aggregation_function_seed_wise : function,
        Function to aggregate the scores returned by all the different Lexicons associated to the seedwords, for a specific token (default=np.max)
    aggregation_function_email_wise : function,
        Function to aggregate the scores given to the tokens in a e-mail (default=np.percentile(.,60))

    Examples
    --------

    """

    def __init__(self, base_seed_words, tokens_column, n_jobs=1,
                 progress_bar=False,
                 extend_seed_word_list=False,
                 normalize_lexicon_scores=False,
                 aggregation_function_seed_wise=np.max,
                 aggregation_function_email_wise=aggregation_percentile_60
                 ):
        """
        Parameters
        ----------
        base_seed_words : list,
            Seedwords list containing the seedwords for computing the Lexicons given by the User.
        tokens_column : str,
            Name of the column in the Pandas Dataframe on which the polarity scores will be computed. Must be a column of tokens.
        n_jobs : int,
            Number of CPUs to use to rate the emails.
        progress_bar : bool,
            Whether to print or not the progress bar while rating the emails.
        extend_seed_word_list : bool,
            Whether to use the seedwords as prefixes.
        normalize_scores : bool,
            Whether or not to normalize the lexicons' scores (so that they are centered around 0 and with a variance of 1)
        aggregation_function_seed_wise : function,
            Function to aggregate the scores returned by all the different Lexicons associated to the seedwords, for a specific token (default=np.max)
        aggregation_function_email_wise : function,
            Function to aggregate the scores given to the tokens in a e-mail (default=np.percentile(.,60))
        """

        self.n_jobs = n_jobs
        self.progress_bar = progress_bar

        self.base_seed_words = base_seed_words
        self.seed_dict = {word: [] for word in self.base_seed_words}
        self.seed_list = base_seed_words
        self.extend_seed_word_list = extend_seed_word_list
        self.tokens_column = tokens_column
        self.normalize_lexicon_scores = normalize_lexicon_scores

        self.lexicon_dict = {}
        self.normalized_lexicon_dict = {}

        self.aggregation_function_seed_wise = aggregation_function_seed_wise
        self.aggregation_function_email_wise = aggregation_function_email_wise

    def __getstate__(self):
        """should return a dict of attributes that will be pickled
        To override the default pickling behavior and
        avoid the pickling of the logger
        """
        d = self.__dict__.copy()
        # disable multiprocessing when saving
        d['n_jobs'] = 1
        d['progress_bar'] = False
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        """To override the default pickling behavior and
        avoid the pickling of the logger"""
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

    def fit(self, embedding):
        """
        Computes the Lexicons for the specific embedding.
        Parameters
        -------
        embedding : Embedding Object,
            A Melusine Embedding object.

        """

        if self.extend_seed_word_list:
            self.seed_dict, self.seed_list = self.compute_seeds_from_root(embedding, self.base_seed_words)

        self.seed_list = [token for token in self.seed_list if token in embedding.embedding.vocab.keys()]

        if not self.seed_list:
            raise ValueError('None of the seed words are in the vocabulary associated with the Embedding')

        self.lexicon_dict = self.compute_lexicon(embedding, self.seed_list)

        if self.normalize_lexicon_scores:
            self.normalize_lexicon()

    @staticmethod
    def compute_seeds_from_root(embedding, base_seed_words):
        """
        Uses the seedwords list provided by the User and treats them as prefixes to find the effective tokens that will be used to compute the Lexicons.

        Parameters
        --------
        embedding : Embedding Object,
            A Melusine Embedding object.
        base_seed_words :list,
            Seedwords list containing the seedwords for computing the Lexicons given by the User.

        Returns
        -------
        (seed_dict, seed_list) : (dict, list),
            Tuple contraining a dict with key:prefixe value : list of seedwords, and a list containing all the seedwords found with the given prefixes

        """
        words = list(embedding.embedding.vocab.keys())
        seed_dict = dict()
        seed_list = []

        for seed in base_seed_words:
            extended_seed_words = [token for token in words if token.startswith(seed)]
            seed_dict[seed] = extended_seed_words
            seed_list.extend(extended_seed_words)

        return seed_dict, seed_list

    @staticmethod
    def compute_lexicon(embedding, seed_list):
        """
        Computes the Lexicons for the given embedding. Computes the cosine similarity between all the tokens in seed_list and the embedding's vocabulary.

        Parameters
        --------
        embedding : Embedding Object,
            A Melusine Embedding object on which the cosine similarity between the words will be computed.
        seed_list : list,
            A list containing the seedwords on which the cosine similarity will be computed.

        Returns
        -------
        lexicon_dict : dict,
            A dict representing the lexicon. The keys will be all the tokens in seed_list, the values will be a dict which keys will be all the tokens of the embedding's vocabulary, and the values their cosine similarity with the seed.

        """

        if type(embedding) == Word2Vec:
            embedding = embedding.wv

        words = list(embedding.embedding.vocab.keys())
        lexicon_dict = {}

        for seed in seed_list:
            lexicon_dict[seed] = {}
            for word in words:
                lexicon_dict[seed][word] = embedding.embedding.similarity(seed, word)

        return lexicon_dict

    def predict(self, X, return_column='score'):
        """
        Given the objet has already been fitted, will add a new column "score"
        (or the column name specified as argument) to the Pandas Dataset containing the polarity scores of the
        documents towards the list of seeds provided.
        Parameters
        ----------
        X : DataFrame
            Input emails DataFrame
        return_column : str
            Name of the new column added to the DataFrame containing the semantic score

        """
        X[return_column] = apply_by_multiprocessing(X, self.rate_email, workers=self.n_jobs,
                                                    progress_bar=self.progress_bar)

        return X

    def normalize_lexicon(self) :
        """
        Normalizes the Lexicon scores (centered around 0, with variance 1).
        """
        lexicon_dict = self.lexicon_dict

        normalized_lexicon=dict()
        for seed in lexicon_dict.keys() :
            mean=np.mean(list(lexicon_dict[seed].values()))
            sd=np.std(list(lexicon_dict[seed].values()))
            lex_norm={k:(v-mean)/sd for k,v in lexicon_dict[seed].items()}
            normalized_lexicon[seed]=lex_norm

        self.normalized_lexicon_dict = normalized_lexicon

    def rate_email(self, row):
        """
        Given the object has been fitted, will compute the polarity score towards the seedwords for a specific e-mail.

        Parameters
        ----------
        row : row,
            A Pandas Dataframe containing a tokenized document.
        """

        # TODO make the aggregation function as an argument
        tokens_column = self.tokens_column
        seed_list = self.seed_list

        if self.normalize_lexicon_scores:
            lexicon_dict = self.normalized_lexicon_dict
        else:
            lexicon_dict = self.lexicon_dict

        effective_tokens_list = [token for token in row[tokens_column] if token in lexicon_dict[seed_list[0]]]

        if effective_tokens_list:

            token_score_list = [
                self.aggregation_function_seed_wise(
                    [lexicon_dict[seed][token] for seed in seed_list]
                )
                for token in effective_tokens_list
            ]

            return self.aggregation_function_email_wise(token_score_list)
        else :
            return(np.nan)
