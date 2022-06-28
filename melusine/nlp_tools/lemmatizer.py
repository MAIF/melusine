import logging
import os
from typing import Union

import numpy
import spacy
import spacy_lefff
from spacy.language import Language
from spacy_lefff import LefffLemmatizer, POSTagger, Downloader

from melusine.utils.verbitim_transformer import VerbitimTransformer

logger = logging.getLogger(__name__)

@Language.factory('melt_tagger')
def create_melt_tagger(nlp, name: str) -> spacy_lefff.POSTagger:
    """
    Instanciates spacy pipeline component melt_tagger
    The decorator @Language.factory('melt_tagger') informs spacy that this is a nlp pipeline component labeled
    'melt_tagger'
    Parameters
    ----------
    nlp : spacy.lang.fr.Language
    A spacy nlp pipe object
    name : str
    Label of the component

    Returns a POSTagger component
    -------

    """
    return POSTagger()


@Language.factory('lefff_lemmatizer')
def create_french_lemmatizer(nlp, name: str) -> spacy_lefff.LefffLemmatizer :
    """
    Instanciates spacy pipeline component lefff_lemmatizer
    The decorator @Language.factory('lefff_lemmatizer') informs spacy that this is a nlp pipeline component labeled
    'lefff_lemmatizer'
    Parameters
    ----------
    nlp : spacy.lang.fr.Language
    A spacy nlp pipe object
    name : str
    Label of the component

    Returns a LefffLemmatizer component
    -------

    """
    return LefffLemmatizer(after_melt=True, default=True)


class BadConfigurationError(Exception):
    """Exception to raise when a bad configuration was passed to lemmatization engine"""
    pass


class Lemmatizer(VerbitimTransformer):
    """ Lemmatizer component for Socle NLP
        Available engines include : spacy and Lefff
    """

    _AVAILABLE_ENGINES = ['spacy',
                          'Lefff'
                          ]

    _SPACY_ARGS = {'exclude': ['senter', 'ner']}

    def __init__(self,
                 in_col: str,
                 out_col: str,
                 engine: str,
                 engine_conf: str,
                 **kwargs
                 ):
        """
        Initialize a Socle NLP Lemmatizer object

        Parameters
        ----------
        in_col : str
        Input column in pandas.DataFrame
        out_col : str
        Output column in pandas.DataFrame.Will be created if it does not exist
        engine : str
        Lemmatization engine to load. Choices are 'spacy' and 'Lefff'.
        engine_conf : str
        Spacy model to load under the hood. For french, choices are 'fr_core_news_sm', 'fr_core_news_md',
        'fr_core_news_lg'.
        """

        self.in_col = in_col
        self.out_col = out_col
        self.engine = engine
        self.engine_conf = engine_conf
        self.tagger = None

        assert engine in self._AVAILABLE_ENGINES, 'Specified engine is not supported. Available engines are ' \
                                                  'spacy, Lefff'
        try:
            if self.engine == 'spacy':
                assert isinstance(self.engine_conf, str), "You have specified spacy as lemmatization engine but " \
                                                          "failed to provide the spacy model name as str"
                self._SPACY_ARGS['name'] = self.engine_conf
                self.nlp = spacy.load(**self._SPACY_ARGS)

            if self.engine == 'Lefff':
                assert isinstance(self.engine_conf, str), "You have specified lefff as lemmatization engine but " \
                                                          "failed to provide the spacy model name as str"
                self._SPACY_ARGS['name'] = self.engine_conf
                self.nlp = spacy.load(**self._SPACY_ARGS)
                self.nlp.add_pipe('melt_tagger', after='parser')
                self.nlp.add_pipe('lefff_lemmatizer', after='melt_tagger')

        except IOError as SpacyError:
            logger.error(SpacyError)
            raise BadConfigurationError('Incorrect Spacy configuration. Please refer to the tutorial')

        def _get_lemmas(text: numpy.ndarray) -> numpy.ndarray:
            lemmas = numpy.empty(shape=text.shape, dtype='object')
            if self.engine == 'spacy':
                for i, doc in enumerate(self.nlp.pipe(text, **kwargs)):
                    lemmas.put(i, ' '.join([token.lemma_ for token in doc]))
            if self.engine == 'Lefff':
                for i, doc in enumerate(self.nlp.pipe(text, **kwargs)):
                    lemmas.put(i, ' '.join([token._.lefff_lemma for token in doc]))
            return lemmas

        super().__init__(_get_lemmas, None, self.in_col, self.out_col)
