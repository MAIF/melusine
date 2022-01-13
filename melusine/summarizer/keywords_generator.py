import copy
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from melusine.utils.transformer_scheduler import TransformerScheduler
from melusine import config

keywords = config["words_list"]["keywords"]
stopwords = config["tokenizer"]["stopwords"] + list(
    config["token_flagger"]["token_flags"]["flag_name"]
)


class KeywordsGenerator(BaseEstimator, TransformerMixin):
    """Class to extract list of keywords from text.

    It is compatible with scikit-learn API (i.e. contains fit, transform
    methods).

    Parameters
    ----------

    max_tfidf_features : int, optional
        Size of vocabulary for tfidf.
        Default value, 10000.

    keywords : list, optional
        Keywords to extracted as priority.
        Default value, "keywords" list defined in conf file.

    stopwords : list, optional
        Stopwords not to be extracted.
        Default value, "names" and "stopwords" lists defined in conf file.

    resample : bool, optional
        True if dataset must be resampled according to class distribution,
        else False.
        Default value, True.

    n_jobs : int, optional
        Number of cores used for computation.
        Default value, 20.

    copy : bool, optional
        Make a copy of DataFrame.
        Default value, True.

    n_max_keywords : int, optional
        Maximum number of keywords to be returned.
        Default value, 6.

    n_min_keywords : int, optional
        Minimum number of keywords to be returned.
        Default value, 0.

    threshold_keywords : float, optional
        Minimum tf-idf score for word to be selected as keyword.
        Default value, 0.0.

    n_docs_in_class : int, optional
        Number of documents in each classes.
        Default value, 100.

    keywords_coef : int, optional
        Coefficient multiplied with the tf-idf scores of each keywords.
        Default value, 10.

    Attributes
    ----------
    max_tfidf_features, keywords, stopwords, resample, n_jobs, progress_bar,
    copy, n_max_keywords, n_min_keywords, threshold_keywords, n_docs_in_class,
    keywords_coef,

    tfidf_vectorizer : TfidfVectorizer instance from sklearn,

    dict_scores_ : dictionary,
        Tf-idf scores for each tokens.

    max_score_ : np.array,

    Examples
    --------
    >>> from melusine.summarizer.keywords_generator import KeywordsGenerator
    >>> keywords_generator = KeywordsGenerator()
    >>> keywords_generator.fit(X, y)
    >>> keywords_generator.transform(X)
    >>> print(X['keywords'])

    """

    def __init__(
        self,
        max_tfidf_features=10000,
        keywords=keywords,
        stopwords=stopwords,
        resample=False,
        n_jobs=1,
        progress_bar=False,
        copy=True,
        n_max_keywords=6,
        n_min_keywords=0,
        threshold_keywords=0.0,
        n_docs_in_class=100,
        keywords_coef=10,
    ):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
        self.keywords = keywords
        self.stopwords = stopwords
        self.resample = resample
        self.n_jobs = n_jobs
        self.progress_bar = progress_bar
        self.copy = copy
        self.n_max_keywords = n_max_keywords
        self.n_min_keywords = n_min_keywords
        self.threshold_keywords = threshold_keywords
        self.n_docs_in_class = n_docs_in_class
        self.keywords_coef = keywords_coef

    def fit(self, X, y=None):
        """Fit the weighted tf-idf model with input data.

        If resample attribute is True the dataset will be resampled according
        to class distribution.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            X must contain ['tokens'] column.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if isinstance(X, dict):
            raise TypeError(
                "You should not use fit on a dictionary object. Use a DataFrame"
            )

        if self.resample:
            X_resample = self.resample_docs(X, y)
        else:
            X_resample = X

        X_resample["tokens"] = X_resample["tokens"].apply(self._remove_stopwords)

        # fit tf-idf on resample data set
        tokens_joined = X_resample["tokens"].apply(lambda x: " ".join(x))
        self.tfidf_vectorizer.fit(tokens_joined)

        # modify the idf weights given frequency in the corpus
        idf_weights = self._add_tf_to_idf(X_resample)
        self.tfidf_vectorizer._tfidf._idf_diag = sp.spdiags(
            idf_weights, diags=0, m=len(idf_weights), n=len(idf_weights)
        )

        # return vetorizer with binary term frequency atribute
        self.dict_scores_ = dict(
            zip(
                self.tfidf_vectorizer.get_feature_names(),
                self.tfidf_vectorizer.idf_,
            )
        )
        self.max_score_ = np.max(self.tfidf_vectorizer.idf_)

        return self

    def transform(self, X):
        """Returns list of keywords in apparition order for each document
        with the weighted tf-idf already fitted.

        Parameters
        ----------
        X : pandas.DataFrame, shape (n_samples, n_features)
            X must contain ['tokens'] column.

        Returns
        -------
        X_new : pandas.DataFrame, shape (n_samples, n_components)
        """
        # Case input is a dict
        if isinstance(X, dict):
            if self.copy:
                X_ = copy.deepcopy(X)
            else:
                X_ = X

            apply_func = TransformerScheduler.apply_dict

        # Case input is a DataFrame
        else:
            if self.copy:
                X_ = X.copy()
            else:
                X_ = X

            apply_func = TransformerScheduler.apply_pandas_multiprocessing

        X_["keywords"] = apply_func(
            X_,
            self.get_keywords,
            args_=None,
            cols_=None,
            n_jobs=self.n_jobs,
            progress_bar=self.progress_bar,
        )

        return X_

    def get_keywords(self, row):
        """Returns list of keywords in apparition order with the
        weighted tf-idf already fitted.

        Parameters
        ----------
        row : row of pd.Dataframe, columns ['tokens']

        Returns
        -------
        list of strings
        """
        tokens = self._remove_stopwords(row["tokens"])
        tokens = [x for x in tokens if not x.isdigit()]
        scores = Counter({t: self.dict_scores_.get(t, 0) for t in tokens})
        n = sum(i > self.threshold_keywords for i in list(scores.values()))
        n = min(n, self.n_max_keywords)
        n = max(n, self.n_min_keywords)
        keywords = [x[0] for x in scores.most_common(n)]
        index_sorted = [(k, tokens.index(k)) for k in keywords if k in tokens]
        index_sorted = sorted(index_sorted, key=lambda x: x[1])
        keywords_sorted = [i[0] for i in index_sorted]

        return keywords_sorted

    def resample_docs(self, X, y=None):
        """Method for resampling documents according to class distribution."""
        X_ = X.copy()
        if y is not None:
            X_["label"] = y
        X_["split"] = 0
        for c in X_.label.unique():
            N_c = X_[X_["label"] == c].shape[0]
            I_c = np.random.randint(0, self.n_docs_in_class + 1, N_c)
            X_.loc[X_["label"] == c, "split"] = I_c

        X_resample = pd.DataFrame(
            X_[["label", "split", "tokens"]]
            .groupby(["label", "split"], as_index=False)["tokens"]
            .sum()
        )

        return X_resample

    def _remove_stopwords(self, tokens):
        """Method to filter stopwords from potential list of keywords."""
        return [t for t in tokens if t not in self.stopwords]

    def _add_tf_to_idf(self, X):
        """Returns the tf-idf weights of each tokens"""
        tokens_joined = X["tokens"].apply(lambda x: " ".join(x))
        X_vec = self.tfidf_vectorizer.transform(tokens_joined)
        feature_names = self.tfidf_vectorizer.get_feature_names()
        idf_weights = self._get_weights(X_vec.toarray(), self.keywords, feature_names)

        return idf_weights

    def _get_weights(self, X_vec, keywords_list, feature_names):
        """Put max weights for each word of redistributed mails."""
        max_ = np.max(X_vec, axis=0)
        mmax_ = np.max(max_)
        for k in keywords_list:
            if k in feature_names:
                max_[feature_names.index(k)] = mmax_ * self.keywords_coef

        return max_
