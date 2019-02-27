import re
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin


class MetaExtension(BaseEstimator, TransformerMixin):
    """Transformer which creates 'extension' feature extracted
    from regex in metadata. It extracts extension of mail adresses.

    Compatible with scikit-learn API.
    """

    def __init__(self):
        self.le_extension = preprocessing.LabelEncoder()

    def fit(self, X, y=None):
        """ Fit LabelEncoder on encoded extensions."""
        X['extension'] = X['from'].apply(self.get_extension)
        self.top_extension = self.get_top_extension(X, n=100)
        X['extension'] = X['extension'].apply(
            lambda x: self.encode_extension(x, self.top_extension))
        self.le_extension.fit(X['extension'])
        return self

    def transform(self, X):
        """Encode extensions"""
        X['extension'] = X['from'].apply(self.get_extension)
        X['extension'] = X['extension'].apply(
            lambda x: self.encode_extension(x, self.top_extension))
        X['extension'] = self.le_extension.transform(X['extension'])
        return X

    @staticmethod
    def get_extension(x):
        """Gets extension from email address."""
        try:
            extension = re.findall(r'\@([^.]+)', x)[0]
        except Exception as e:
            return ''
        return extension

    @staticmethod
    def get_top_extension(X, n=100):
        "Returns list of most common extensions."
        a = Counter(X['extension'].values)
        a = a.most_common(n)
        a = [x[0] for x in a]
        return a

    @staticmethod
    def encode_extension(x, top_ext):
        """Encode most common extensions and set the rest to 'other'."""
        if x in top_ext:
            return x
        else:
            return 'other'


class MetaDate(BaseEstimator, TransformerMixin):
    """Transformer which creates new features from dates such as:
        - hour
        - minute
        - dayofweek

    Compatible with scikit-learn API.

    Parameters
    ----------
    date_format : str, optional
        Regex to extract date from text.

    date_format : str, optional
        A date format.
    """

    def __init__(self,
                 regex_date_format=r'\w+ (\d+) (\w+) (\d{4}) (\d{2}) h (\d{2})',
                 date_format='%d/%m/%Y %H:%M'):
        self.regex_date_format = regex_date_format
        self.date_format = date_format
        self.month = {
            'janvier': '1',
            'février': '2',
            'mars': '3',
            'avril': '4',
            'mai': '5',
            'juin': '6',
            'juillet': '7',
            'août': '8',
            'septembre': '9',
            'octobre': '10',
            'novembre': '11',
            'décembre': '12',
        }

    def fit(self, X, y=None):
        """Unused method. Defined only for compatibility with scikit-learn API.
        """
        return self

    def transform(self, X):
        """Transform date to hour, min, day features."""
        X['date'] = X['date'].apply(self.date_formatting,
                                    args=(self.regex_date_format, ))
        X['date'] = pd.to_datetime(X['date'],
                                   format=self.date_format,
                                   infer_datetime_format=False,
                                   errors='coerce')
        X['hour'] = X['date'].apply(self.get_hour)
        X['min'] = X['date'].apply(self.get_min)
        X['dayofweek'] = X['date'].apply(self.get_dayofweek)
        return X

    def date_formatting(self, x, regex_format):
        """Set a date in the right format"""
        try:
            e = re.findall(regex_format, x)[0]
            date = e[0]+'/'+e[1]+'/'+e[2]+' '+e[3]+':'+e[4]
            for m, m_n in self.month.items():
                date = date.replace(m, m_n)
        except Exception as e:
            return x
        return date

    @staticmethod
    def get_hour(x):
        """Get hour from date"""
        try:
            return x.hour
        except Exception as e:
            return 0

    @staticmethod
    def get_min(x):
        """Get minutes from date"""
        try:
            return x.minute
        except Exception as e:
            return 0

    @staticmethod
    def get_dayofweek(x):
        """Get day of the week from date"""
        try:
            return x.dayofweek
        except Exception as e:
            return 0


class Dummifier(BaseEstimator, TransformerMixin):
    """Transformer to dummifies categorial features.
    Compatible with scikit-learn API.
    """

    def __init__(self,
                 columns_to_dummify=['extension', 'dayofweek', 'hour', 'min'],
                 copy=True):
        self.columns_to_dummify = columns_to_dummify
        self.copy = copy
        pass

    def fit(self, X, y=None):
        """Store dummified features to avoid inconsistance of
        new data which could contain new labels (unknown from train data).
        """
        self.X_ = pd.get_dummies(
            X, columns=self.columns_to_dummify, prefix_sep='__', dummy_na=False)

        dummies_ = tuple([col + '__' for col in self.columns_to_dummify])
        self.dummy_features = [c for c in self.X_.columns if c.startswith(dummies_)]

        return self

    def transform(self, X, y=None):
        """Dummify features and keep only common labels with pretrained data.
        """
        if self.copy:
            X_ = X.copy()
        else:
            X_ = X

        X_ = pd.get_dummies(
            X_, columns=self.columns_to_dummify, prefix_sep='__', dummy_na=False)

        return X_[self.dummy_features]
