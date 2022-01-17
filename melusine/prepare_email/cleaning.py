"""
Cleaning of the body and the header
"""

import unidecode
import unicodedata
import re
from melusine import config

REGEX_CLEAN = config["regex"]["cleaning"]
regex_flags_dict = config["text_flagger"]["text_flags"]
regex_clean_header_dict = REGEX_CLEAN["clean_header_dict"]
regex_remove_multiple_spaces_list = REGEX_CLEAN["remove_multiple_spaces_list"]


def clean_body(row, flags=True):
    """Clean body column. The cleaning involves the following operations:
        - Cleaning the text
        - Removing the multiple spaces
        - Flagging specific items (postal code, phone number, date...)

    Parameters
    ----------
    row : row of pandas.Dataframe object,
        Data contains 'last_body' column.

    flags : boolean, optional
        True if you want to flag relevant info, False if not.
        Default value, True.

    Returns
    -------
    row of pandas.DataFrame object or pandas.Series if apply to all DF.
    """
    text = str(row["last_body"])
    clean_body = clean_text(text)
    clean_body = flag_items(clean_body, flags=flags)
    return clean_body


def clean_header(row, flags=True):
    """Clean the header column. The cleaning involves the following operations:
        - Removing the transfers and answers indicators
        - Cleaning the text
        - Flagging specific items (postal code, phone number, date...)

    Parameters
    ----------
    row : row of pandas.Dataframe object,
        Data contains 'header' column.

    flags : boolean, optional
        True if you want to flag relevant info, False if not.
        Default value, True.

    Returns
    -------
    row of pd.DataFrame object or pandas.Series if apply to all DF.
    """
    text = str(row["header"])
    clean_header = remove_transfer_answer_header(text)
    clean_header = clean_text(clean_header)
    clean_header = flag_items(clean_header, flags=flags)
    return clean_header


def clean_text(text):
    """Clean a string. The cleaning involves the following operations:
        - Putting all letters to lowercase
        - Removing all the accents
        - Removing all line breaks
        - Removing all symbols and punctuations
        - Removing the multiple spaces

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    text = text_to_lowercase(text)
    text = remove_accents(text)
    text = remove_line_break(text)
    text = remove_superior_symbol(text)
    # text = remove_apostrophe(text)
    text = remove_multiple_spaces_and_strip_text(text)
    return text


def text_to_lowercase(text):
    """Set all letters to lowercase"""
    return text.lower()


def remove_accents(text, use_unidecode=False):
    """
    Remove accents from text
    Using unidecode is more powerful but much more time consuming
    Exemple: the joined 'ae' character is converted to 'a' + 'e' by unidecode while it is suppressed by unicodedata.

    """
    if use_unidecode:
        return unidecode.unidecode(text)
    else:
        utf8_str = (
            unicodedata.normalize("NFKD", text)
            .encode("ASCII", "ignore")
            .decode("utf-8")
        )
        return utf8_str


def remove_line_break(text):
    """Remove line breaks from text"""
    return text.replace("\n", "")


def remove_superior_symbol(text):
    """Remove superior and inferior symbols from text"""
    text = text.replace(">", "")
    text = text.replace("<", "")
    return text


def remove_apostrophe(text):
    """Remove apostrophes from text"""
    return text.replace("'", " ")


def remove_multiple_spaces_and_strip_text(text):
    """Remove multiple spaces, strip text, and remove '-', '*' characters.

    Parameters
    ----------
    text : str,
        Header content.

    Returns
    -------
    str

    """
    for regex_remove_multiple_spaces in regex_remove_multiple_spaces_list:
        text = re.sub(regex_remove_multiple_spaces, " ", text)
        text = text.strip()
    return text


def flag_items(text, flags=True):
    """Flag relevant information
        ex : amount, phone number, email address, postal code (5 digits)..

    Parameters
    ----------
    text : str,
        Body content.

    flags : boolean, optional
        True if you want to flag relevant info, False if not.
        Default value, True.

    Returns
    -------
    str

    """
    if flags:
        for regex, value in regex_flags_dict.items():
            text = re.sub(pattern=regex, repl=value, string=text, flags=re.IGNORECASE)
        return text
    else:
        return text


def remove_transfer_answer_header(text):
    """Remove historic and transfers indicators in the header.
    Ex: "Tr:", "Re:", "Fwd", etc.

    Parameters
    ----------
    text : str,
        Header content.

    Returns
    -------
    str

    """
    for regex, value in regex_clean_header_dict.items():
        text = re.sub(pattern=regex, repl=value, string=text, flags=re.IGNORECASE)
    return text
