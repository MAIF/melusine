import re
import string
import unidecode
import unicodedata
from typing import List
from collections import Counter
from pandas.core.series import Series
from socle_nlp.utils.socle_nlp_transformer import SocleNlpTransformer


class ReplacePatternTransformer(SocleNlpTransformer):

    def __init__(self, in_col: str, out_col: str, repl_pattern_regex: List[str], by: str):
        if isinstance(repl_pattern_regex, list):
            repl_pattern_regex = '|'.join(repl_pattern_regex)

        self.repl_pattern_regex = re.compile(repl_pattern_regex)

        def __replace_pattern(s: Series, by_sub: str) -> Series:
            return s.str.replace(self.repl_pattern_regex, by_sub, regex=True)

        super().__init__(__replace_pattern, {'by_sub': by}, in_col, out_col)


class RemovePatternTransformer(ReplacePatternTransformer):
    def __init__(self, in_col: str, out_col: str, rm_pattern_regex: List[str]):
        super().__init__(in_col, out_col, rm_pattern_regex, "")


class RemoveMultiSpaceTransformer(ReplacePatternTransformer):
    def __init__(self, in_col: str, out_col: str):
        regex = r"\t|\s{2,}"
        super().__init__(in_col, out_col, [regex], ' ')


class RemovePunctTransformer(RemovePatternTransformer):
    def __init__(self, in_col: str, out_col: str):
        regex = r'[%s]' % re.escape(string.punctuation)
        super().__init__(in_col, out_col, [regex])


class RemoveLineBreakTransformer(RemovePatternTransformer):
    def __init__(self, in_col: str, out_col: str):
        regex = r"\n|\r"
        super().__init__(in_col, out_col, [regex])


class StripTransformer(RemovePatternTransformer):
    def __init__(self, in_col: str, out_col: str):
        regex = r"^[\t\s]+|[\t\s]+$"
        super().__init__(in_col, out_col, [regex])


class ToStrTransformer(SocleNlpTransformer):
    def __init__(self, in_col: str, out_col: str):
        def __to_str_fun(s: Series) -> Series:
            return s.astype(str)

        super().__init__(__to_str_fun, None, in_col, out_col)


class ToLowerTransformer(SocleNlpTransformer):
    def __init__(self, in_col: str, out_col: str):
        def __to_lower(s: Series) -> Series:
            return s.str.lower()

        super().__init__(__to_lower, None, in_col, out_col)


class RemoveAccentTransformer(SocleNlpTransformer):
    def __init__(self, in_col: str, out_col: str):

        def __rem_accent(text: str, use_unidecode: bool = False) -> str:
            if use_unidecode:
                return unidecode.unidecode(text)
            else:
                utf8_str = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
                return utf8_str

        def __remove_accent(s: Series) -> Series:
            return s.apply(lambda x: __rem_accent(x))

        super().__init__(__remove_accent, None, in_col, out_col)


class StopWordsTransformer(SocleNlpTransformer):
    def __init__(self, in_col: str, out_col: str, stopwords: List[str]):
        self.__stopwords = Counter(stopwords)

        def __rm_stopwords(s: Series) -> Series:
            return s.apply(lambda x: ' '.join([word for word in x.split() if word not in self.__stopwords]))

        super().__init__(__rm_stopwords, None, in_col, out_col)
