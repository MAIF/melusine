import logging
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from socle_nlp.text_processing.all_transformers import *


def _check_tuple_transformer(transformer_tuple: Tuple) -> Tuple:
    if len(transformer_tuple) != 2:
        raise TypeError(f'Tuple must be (name, SocleNlpTransformer)')
    name, transformer = transformer_tuple
    if not isinstance(name, str):
        raise TypeError(f'Tuple must be (name, SocleNlpTransformer) with name a str')
    if not issubclass(transformer.__class__, SocleNlpTransformer):
        raise TypeError(f'Tuple must be (name, SocleNlpTransformer) with SocleNlpTransformer \
            a subclass of SocleNlpTransformer')
    return transformer_tuple


class SocleNlpPipeline(BaseEstimator, TransformerMixin):

    def __init__(self, sequence: List):

        self.sequence = sequence
        self.logger = logging.getLogger(__name__)
        
        def _create_pipeline_from_sequence(transformers_seq) -> Pipeline:
            transformers = []
            for trf in transformers_seq:
                if type(trf) == tuple and _check_tuple_transformer(trf):
                    transformers.append(trf)
            return Pipeline(transformers)
        
        self.socle_pipeline = _create_pipeline_from_sequence(sequence)

    def fit(self, X, y=None):
        """Unused method. Defined only for compatibility with scikit-learn API."""
        return self

    def transform(self, X):
        return self.socle_pipeline.fit_transform(X)
