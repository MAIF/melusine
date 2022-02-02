import logging
import os
from typing import Union

import numpy
import spacy
from spacy.language import Language
from spacy_lefff import LefffLemmatizer, POSTagger, Downloader
from treetaggerwrapper import TreeTaggerError

from socle_nlp.utils.socle_nlp_transformer import SocleNlpTransformer


@Language.factory('melt_tagger')
def create_melt_tagger(nlp, name):
    # TODO:Variable de release de l'asset
    return POSTagger()


@Language.factory('lefff_lemmatizer')
def create_french_lemmatizer(nlp, name):
    return LefffLemmatizer(after_melt=True, default=True)


class BadConfigurationError(Exception):
    """Exception to raise when a bad configuration was passed to lemmatization engine"""
    pass


class Lemmatizer(SocleNlpTransformer):
    """ Lemmatizer component for Socle NLP
        Available engines include : spacy, Lefff and TreeTagger
    """

    _AVAILABLE_ENGINES = ['spacy',
                          'Lefff',
                          'TreeTagger']

    _TTW_ARGS = {'TAGLANG': 'fr',
                 'TAGPARFILE': 'french.par'
                 }

    _SPACY_ARGS = {'exclude': ['senter', 'ner']}

    def __init__(self,
                 in_col: str,
                 out_col: str,
                 engine: str,
                 engine_conf: Union[dict, str, None],
                 n_jobs: int = 1,
                 batch_size: int = 1000
                 ):
        """
        Initialize a Socle NLP Lemmatizer object
        :param in_col:  input column of dataframe.
        :param out_col: output column, will be created if it does not exist.
        :param engine: lemmatization engine to load.
        :param engine_conf: configuration for the engine chosen. For TreeTagger, a dictionary containing TAGDIR,
                            TAGLANG, TAGPARFILE keys is required. For spacy based lemmatizers, the name of the model
                            must be provided. Refer to tutorials for more information.
        :param n_jobs: spacy-based engines only. Number of processes to be spawned by spacy pipe.
        :param batch_size: spacy-based engines only. Batch size to process at once in spacy pipe.
        """

        self.in_col = in_col
        self.out_col = out_col
        self.engine = engine
        self.engine_conf = engine_conf
        self.n_jobs = n_jobs
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)
        self.tagger = None

        assert engine in self._AVAILABLE_ENGINES, 'Specified engine is not supported. Available engines are ' \
                                                  'spacy, Lefff and TreeTagger'
        try:
            if self.engine == 'TreeTagger':
                import treetaggerwrapper as ttw
                try:
                    self.tagger = ttw.TreeTagger(**self.engine_conf)
                except TreeTaggerError as tte:
                    self.logger.error("TreeTagger archive path incorrect")
                    raise TreeTaggerError('Please download TreeTagger archive here and specify extraction location as TAGDIR \
                                            : https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/')

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
            self.logger.error(SpacyError)
            raise BadConfigurationError('Incorrect Spacy configuration. Please refer to the tutorial')
        except TreeTaggerError as tte:
            self.logger.error(tte)
            raise BadConfigurationError('Incorrect TreeTagger configuration. Please refer to the tutorial')

        def _get_lemmas(text: numpy.ndarray) -> numpy.ndarray:
            lemmas = numpy.empty(shape=text.shape, dtype='object')
            if self.engine == 'TreeTagger':
                lines = map(self.tagger.tag_text, text)
                for i, tags in enumerate(lines):
                    lemmas.put(i, ' '.join([tag.split('\t')[-1] for tag in tags]))
            if self.engine == 'spacy':
                for i, doc in enumerate(self.nlp.pipe(text, n_process=self.n_jobs, batch_size=self.batch_size)):
                    lemmas.put(i, ' '.join([token.lemma_ for token in doc]))
            if self.engine == 'Lefff':
                for i, doc in enumerate(self.nlp.pipe(text, n_process=self.n_jobs, batch_size=self.batch_size)):
                    lemmas.put(i, ' '.join([token._.lefff_lemma for token in doc]))
            return lemmas

        super().__init__(_get_lemmas, None, self.in_col, self.out_col)