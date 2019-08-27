import logging
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool

from sklearn.base import BaseEstimator, TransformerMixin
from melusine.utils.multiprocessing import apply_by_multiprocessing


class SentimentDetector(BaseEstimator, TransformerMixin):

    def __init__(self, base_seed_words, tokens_column, n_jobs=1,
                 progress_bar=False, root=False,
                 normalize_lexicon=False,
                 aggregation_function_seed_wise=np.max,
                 aggregation_function_email_wise=lambda x: np.percentile(x, 60)
                 ):
        """
        TODO

        :param n_jobs:
        :param progress_bar:
        """

        self.n_jobs = n_jobs
        self.progress_bar = progress_bar

        self.base_seed_words = base_seed_words
        self.seed_dict = {word: [] for word in self.base_seed_words}
        self.seed_list = base_seed_words
        self.root = root
        self.tokens_column = tokens_column
        self.normalize_lexicon = normalize_lexicon

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
        Not used

        Returns
        -------

        """

        if self.root:
            self.seed_dict, self.seed_list = self.compute_seeds_from_root(embedding, self.base_seed_words)

        self.lexicon_dict = self.compute_lexicon(embedding, self.seed_list)

    @staticmethod
    def compute_seeds_from_root(embedding, base_seed_words):
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
        words = list(embedding.embedding.vocab.keys())
        lexicon_dict = {}

        for seed in seed_list:
            lexicon_dict[seed] = {}
            for word in words:
                lexicon_dict[seed][word] = embedding.embedding.similarity(seed, word)

        return lexicon_dict


    def predict(self, X):
        """

        Parameters
        ----------
        X : DataFrame
            Input emails DataFrame

        Returns
        -------

        """
        X['sentiment_score'] = apply_by_multiprocessing(X, self.rate_email, workers=self.n_jobs,
                                             progress_bar=self.progress_bar)

        return X

    def normalize_lexicon(self) :
        lexicon_dict = self.lexicon_dict

        normalized_lexicon=dict()
        for seed in lexicon_dict.keys() :
            mean=np.mean(list(lexicon_dict[seed].values()))
            sd=np.std(list(pols[seed].values()))
            lex_norm={k:(v-mean)/sd for k,v in lexicon_dict[seed].items()}
            normalized_lexicon[seed]=lex_norm

        self.normalized_lexicon_dict = normalized_lexicon

    def rate_email(self, row):

        # TODO make the aggregation function as an argument
        tokens_column = self.tokens_column
        seed_list = self.seed_list

        if self.normalize_lexicon:
            lexicon_dict = self.normalized_lexicon_dict
        else:
            lexicon_dict = self.lexicon_dict

        effective_tokens_list = [token for token in row[tokens_column] if token in lexicon_dict[seed_list[0]]]

        token_score_list = [
            self.aggregation_function_seed_wise(
                [lexicon_dict[seed][token] for seed in seed_list]
            )
            for token in effective_tokens_list
        ]

        return self.aggregation_function_email_wise(token_score_list)

    def print_topn_mails(mails_rated, mails_raw, n=5, lab=False, rev=True):
        get_mails_idx=list()
        ids=sorted(mails_rated, key=mails_rated.__getitem__, reverse=rev)
        for j in range(n) :
            get_mails_idx.append(ids[j])
        for elem in get_mails_idx :
            print("Index : " + str(mails_raw.index.values[elem]))
            if lab==True :
                print("Label : " + str(mails_raw.iloc[elem].mec_label))
            print("Score : "+ str(mails_rated[elem] ))
            print("\n")
            print("Header : "+str(mails_raw.iloc[elem].conversation_epure[0]['structured_text']["header"]))
            print("Body : " +str(mails_raw.iloc[elem].conversation_epure[0]['structured_text']["text"]))
            print("\n---------------------------------------------------\n")


    def get_parse_results(session_id):
        url = "http://sc100525.maif.local:8099/api/sessions/results/"+str(session_id)
        querystring = {"rasa":"False%20"}
        headers = {'User-Agent': "PostmanRuntime/7.15.0",'Accept': "*/*",'Cache-Control': "no-cache",'Postman-Token': "8fa3962c-8204-48eb-92ef-df1b79678331,1db1abec-b1c9-4ac9-b709-bc7b992c9dc4",'Host': "sc100525.maif.local:8099",'accept-encoding': "gzip, deflate",'Connection': "keep-alive",'cache-control': "no-cache"}
        response = requests.request("GET", url, headers=headers, params=querystring)
        res=json.loads(response.text)
        res_label=dict()
        for email in res :
            id_mail=int(re.search("(?:\|\|)(\d+)(?:\|\|)", email['text'])[1])
            if email["answer"]=='accept':
                label=1
            elif email["answer"]=='reject' :
                label=0
            else :
                label=-1
            res_label[id_mail]=label
        return(res_label)
