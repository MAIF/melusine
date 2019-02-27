import nltk
from melusine.prepare_email.mail_segmenting import split_message_to_sentences
from melusine.utils.multiprocessing import apply_by_multiprocessing


class Streamer():
    """Class to transform pd.Series into stream.

    Used to prepare the data for the training of the phraser and embeddings.

    Attributes
    ----------
    columns : str or list of str,
        Input text column(s) to consider for the streamer.

    stream : MailIterator object,
        Stream of all the tokens of the pd.Series.

    Examples
    --------
    >>> streamer = Streamer()
    >>> streamer.to_stream(X) # will build the stream attribute
    >>> tokens_stream =  = streamer.stream
    >>> print(tokens_stream)

    """

    def __init__(self, columns='clean_body', n_jobs=40):
        self.columns_ = columns
        self.n_jobs = n_jobs

    def to_stream(self, X):
        """Build a MailIterator object containing a stream of tokens from
        a pd.Series.

        Parameters
        ----------
        X : pd.Dataframe.

        Examples
        --------
        >>> streamer.to_stream(X) # will build the stream attribute
        >>> tokens_stream =  = streamer.stream
        >>> print(tokens_stream)

        """
        flattoks = self.to_flattoks(X)
        self.stream = MailIterator(flattoks)
        pass

    def to_flattoks(self, X):
        """Create list of list of tokens from a pd.Series
        Each list of tokens correspond to a sentence.

        Parameters
        ----------
        X : pd.Dataframe,

        Returns
        -------
        list of lists of strings
        """
        tokenized_sentences_list = apply_by_multiprocessing(X[self.columns_],
                                                            self.to_list_of_tokenized_sentences,
                                                            workers=self.n_jobs
                                                            )
        flattoks = [item for sublist in tokenized_sentences_list
                    for item in sublist]
        return flattoks

    def to_list_of_tokenized_sentences(self, text):
        """Create list of list of tokens from a text.
        Each list of tokens correspond to a sentence.

        Parameters
        ----------
        text : str

        Returns
        -------
        list of list of strings
        """
        sentences_list = split_message_to_sentences(text)
        tokenized_sentences_list = [nltk.regexp_tokenize(sentence,
                                                         pattern="\w+(?:[\?\-\'\"_]\w+)*")
                                    for sentence in sentences_list
                                    if sentence != ""]
        return tokenized_sentences_list


class MailIterator():
    """Class to transform stream of tokens into iterators."""

    def __init__(self, tok_stream):
        self.tok_stream = tok_stream

    def __iter__(self):
        for sent in self.tok_stream:
            yield sent
