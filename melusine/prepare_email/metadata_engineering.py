import re
import copy
import pandas as pd
from collections import Counter
from itertools import chain
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from melusine.utils.transformer_scheduler import TransformerScheduler


class MetaExtension(BaseEstimator, TransformerMixin):
    """Transformer which creates 'extension' feature extracted
    from regex in metadata. It extracts extension of mail adresses.

    Compatible with scikit-learn API.
    """

    def __init__(self, topn_extension=100):
        self.le_extension = preprocessing.LabelEncoder()
        self.topn_extension = topn_extension

    def fit(self, X, y=None):

        if isinstance(X, dict):
            raise TypeError(
                "You should not use fit on a dictionary object. Use a DataFrame"
            )

        """ Fit LabelEncoder on encoded extensions."""
        X["extension"] = X.apply(self.get_extension, axis=1)
        self.top_extension = self.get_top_extension(X, n=self.topn_extension)
        X["extension"] = X.apply(
            self.encode_extension, args=(self.top_extension,), axis=1
        )
        # We add the value "other" for the training of the label encoder to deal with unseen values of metadatas during inference !
        self.le_extension.fit(pd.concat((X["extension"], pd.Series(["other"]))))
        return self

    def transform(self, X):
        """Encode extensions"""

        if isinstance(X, dict):
            apply_func = TransformerScheduler.apply_dict
        else:
            apply_func = TransformerScheduler.apply_pandas

        X["extension"] = apply_func(X, self.get_extension)
        X["extension"] = apply_func(
            X, self.encode_extension, args_=(self.top_extension,)
        )
        if isinstance(X["extension"], str):
            X["extension"] = self.le_extension.transform([X["extension"]])[0]
        else:
            X["extension"] = self.le_extension.transform(X["extension"])
        return X

    @staticmethod
    def get_extension(row):
        """Gets extension from email address."""
        x = row["from"]
        try:
            extension = re.findall(r"\@([^.]+)", x)[0]
        except Exception:
            return ""
        return extension

    @staticmethod
    def get_top_extension(X, n=100):
        "Returns list of most common extensions."
        a = Counter(X["extension"].values)
        a = a.most_common(n)
        a = [x[0] for x in a]
        return a

    @staticmethod
    def encode_extension(row, top_ext):
        x = row["extension"]
        """Encode most common extensions and set the rest to 'other'."""
        if x in top_ext:
            return x
        else:
            return "other"


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

    def __init__(
        self,
        regex_date_format=r"\w+ (\d+) (\w+) (\d{4}) (\d{2}) h (\d{2})",
        date_format="%d/%m/%Y %H:%M",
    ):
        self.regex_date_format = regex_date_format
        self.date_format = date_format
        self.month = {
            "janvier": "1",
            "février": "2",
            "mars": "3",
            "avril": "4",
            "mai": "5",
            "juin": "6",
            "juillet": "7",
            "août": "8",
            "septembre": "9",
            "octobre": "10",
            "novembre": "11",
            "décembre": "12",
        }

    def fit(self, X, y=None):
        """Unused method. Defined only for compatibility with scikit-learn API."""
        return self

    def transform(self, X):

        if isinstance(X, dict):
            apply_func = TransformerScheduler.apply_dict
        else:
            apply_func = TransformerScheduler.apply_pandas

        """Transform date to hour, min, day features."""
        X["date"] = apply_func(X, self.date_formatting, args_=(self.regex_date_format,))
        X["hour"] = apply_func(X, self.get_hour)
        X["min"] = apply_func(X, self.get_min)
        X["dayofweek"] = apply_func(X, self.get_dayofweek)
        return X

    def date_formatting(self, row, regex_format):
        """Set a date in the right format"""
        x = row["date"]
        try:
            e = re.findall(regex_format, x)[0]
            date = e[0] + "/" + e[1] + "/" + e[2] + " " + e[3] + ":" + e[4]
            for m, m_n in self.month.items():
                date = date.replace(m, m_n)
            date = pd.to_datetime(
                date,
                format=self.date_format,
                infer_datetime_format=False,
                errors="coerce",
            )
        except Exception:
            return pd.to_datetime(x)
        return date

    @staticmethod
    def get_hour(row):
        """Get hour from date"""
        x = row["date"]
        try:
            return x.hour
        except Exception:
            return 0

    @staticmethod
    def get_min(row):
        x = row["date"]
        """Get minutes from date"""
        try:
            return x.minute
        except Exception:
            return 0

    @staticmethod
    def get_dayofweek(row):
        """Get day of the week from date"""
        x = row["date"]

        try:
            return x.dayofweek
        except Exception:
            return 0


class Dummifier(BaseEstimator, TransformerMixin):
    """Transformer to dummifies categorial features and list of .
    Compatible with scikit-learn API.
    """

    def __init__(
        self,
        columns_to_dummify=["extension", "dayofweek", "hour", "min", "attachment_type"],
        copy=True,
    ):
        self.columns_to_dummify = columns_to_dummify
        self.copy = copy
        pass

    def fit(self, X, y=None):
        """Store dummified features to avoid inconsistance of
        new data which could contain new labels (unknown from train data).
        """

        if isinstance(X, dict):
            raise TypeError(
                "You should not use fit on a dictionary object. Use a DataFrame"
            )

        X_ = pd.get_dummies(
            X,
            columns=[
                col for col in self.columns_to_dummify if col != "attachment_type"
            ],
            prefix_sep="__",
            dummy_na=False,
        )

        dummies_ = tuple([col + "__" for col in self.columns_to_dummify])

        if "attachment_type" in self.columns_to_dummify:
            X_attachment = pd.get_dummies(
                X["attachment_type"].apply(pd.Series).stack().astype(int)
            ).sum(level=0)
            X_attachment = X_attachment.add_prefix("attachment_type__")
            self.dummy_features = [
                c
                for c in pd.concat([X_, X_attachment], axis=1)
                if c.startswith(dummies_)
            ]
        else:
            self.dummy_features = [c for c in X_ if c.startswith(dummies_)]

        return self

    def transform(self, X, y=None):
        """Dummify features and keep only common labels with pretrained data."""
        return_dict = False

        # Case input is a dict
        if isinstance(X, dict):
            if self.copy:
                X_ = copy.deepcopy(X)
            else:
                X_ = X

            X_ = pd.DataFrame([X_])

            return_dict = True

        # Case input is a DataFrame
        else:
            if self.copy:
                X_ = X.copy()
            else:
                X_ = X

        X_ = pd.get_dummies(
            X_,
            columns=[
                col for col in self.columns_to_dummify if col != "attachment_type"
            ],
            prefix_sep="__",
            dummy_na=False,
        )

        if "attachment_type" in self.columns_to_dummify:
            X_attachment = pd.get_dummies(
                X_["attachment_type"].apply(pd.Series).stack().astype(int)
            ).sum(level=0)
            X_attachment = X_attachment.add_prefix("attachment_type__")
            X_ = pd.concat([X_, X_attachment], axis=1)

        if return_dict:
            X_ = X_.T.reindex(self.dummy_features).T.fillna(0)
            return X_[self.dummy_features].to_dict(orient="records")[0]
        else:
            X_ = X_.T.reindex(self.dummy_features).T.fillna(0)
            return X_[self.dummy_features]


class MetaAttachmentType(BaseEstimator, TransformerMixin):
    """Transformer which creates 'type' feature extracted
    from regex in metadata. It extracts types of attached files.

    Compatible with scikit-learn API.
    """

    def __init__(self, topn_extension=100):
        self.le_extension = preprocessing.LabelEncoder()
        self.topn_extension = topn_extension

    def fit(self, X, y=None):

        if isinstance(X, dict):
            raise TypeError(
                "You should not use fit on a dictionary object. Use a DataFrame"
            )

        """ Fit LabelEncoder on encoded extensions."""
        X["attachment_type"] = X.apply(self.get_attachment_type, axis=1)
        self.top_attachment_type = self.get_top_attachment_type(
            X, n=self.topn_extension
        )
        X["attachment_type"] = X.apply(
            self.encode_type, args=(self.top_attachment_type,), axis=1
        )
        # We add the value "other" for the trainign of the label encoder to deal with unseen values during inference !
        self.le_extension.fit(pd.concat((X["attachment_type"], pd.Series([['other']]))).sum())
        return self

    def transform(self, X):
        """Encode extensions"""

        if isinstance(X, dict):
            apply_func = TransformerScheduler.apply_dict
        else:
            apply_func = TransformerScheduler.apply_pandas

        X["attachment_type"] = apply_func(X, self.get_attachment_type)
        X["attachment_type"] = apply_func(
            X, self.encode_type, args_=(self.top_attachment_type,)
        )
        if isinstance(X["attachment_type"], list):
            X["attachment_type"] = self.le_extension.transform([X["attachment_type"]])[
                0
            ]
        else:
            X["attachment_type"] = [
                self.le_extension.transform(t) for t in X["attachment_type"]
            ]
        return X

    @staticmethod
    def get_attachment_type(row):
        """Gets type from attachment."""
        x = row["attachment"]
        attached_types = []

        try:
            for file in x:
                match = re.findall(r".*\.(.*)", file)
                if match:
                    attached_types.append(match[0])
        except Exception:
            return ""
        return attached_types

    @staticmethod
    def get_top_attachment_type(X, n=100):
        "Returns list of most common types of attachment."

        type_counter = Counter(chain(*X["attachment_type"]))
        type_counter = type_counter.most_common(n)
        top_attachment_type = [x[0] for x in type_counter]
        return top_attachment_type

    @staticmethod
    def encode_type(row, top_ext):
        x = row["attachment_type"]
        """Encode most common type of attachment and set the rest to 'other'."""
        encode = []
        if x:
            for attachment in x:
                if attachment in top_ext:
                    encode.append(attachment)
                else:
                    encode.append("other")
        else:  # No attachments
            encode.append("none")
        return encode
