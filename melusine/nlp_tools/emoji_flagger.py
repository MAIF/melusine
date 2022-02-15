import logging

logger = logging.getLogger(__name__)


class DeterministicEmojiFlagger:
    FILENAME = "emoji_flagger.json"

    def __init__(self, input_column: str ="clean_body", output_column: str ="clean_body", flag_emoji: str = " flag_emoji_ "):
        """Apply emoji flagging to string (header, body...)

        Parameters
        ----------
        input_column : str,
            Column of pd.Dataframe which contains a list of tokens, default column ['tokens']
        output_column: str,
            Column where is saved the list of stemmed tokens, default column ['stemmed_tokens']
        flag_emoji : str,
            Flag you want to insert in text
        
        Returns
        -------
        pd.Dataframe
                Examples
        --------
        >>> from melusine.nlp_tools.emoji_flagger import DeterministicEmojiFlagger
        >>> emoji_flagger = DeterministicEmojiFlagger()
        >>> emoji_flagger.transform(data)
        """
        from emoji import get_emoji_regexp
        self.emoji_pattern = get_emoji_regexp()
        self.flag_emoji = flag_emoji

    @staticmethod
    def _flag_emojis(text: str) -> str:
        """Flag emojis
        WARNING : Execution time for flagging emojis is significantly higher than for other flags, which may make it inconvenient on a large sample size of emails, thus emojis aren't flagged by default. 
        Parameters
        ----------
        text : str,
            Body content.
        Returns
        -------
        str
        """
        text = self.emoji_pattern.sub(repl=self.flag_emoji,  string=text)
        return text

    def fit(self, df, y=None):
        """ """
        return self

    def transform(self, df):
        input_data = df[self.input_column]
        df[self.output_column] = input_data.apply(self._flag_emojis)
        return df
