import numpy as np


def structured_score(row):
    """Function to be called after the mail_segmenting function as it requires a "structured_body" column.
    Return a set with the different parts tag (ex : "HELLO", "BODY", etc.) present, or "EMPTY" if any,  in the last
    email of the conversation .

    Parameters
    ----------
    row : pd.Series
        Row of an email DataFrame

    Returns
    -------
    tags_set : a set of the different parts tag found in the email or EMPTY if None
    length of tags_set : count of parts tag found in the email

    """
    parts_type = []
    for part in row["structured_body"][0]["structured_text"]["text"]:
        parts_type.append(part["tags"])
    tags_set = set(parts_type)
    return (tags_set or set(["EMPTY"]), len(tags_set))


def mean_words_by_sentence(row, tokenizer):
    """Function to be called after the mail_segmenting function as it requires a "structured_body" column.
    Compute the average number of words per sentence for body sentences

    Parameters
    ----------
    row : pd.Series
        Row of an email DataFrame

    tokenizer : an object from the Tokenizer melusine class (nlp_tools.tokenizer.Tokenizer)

    Returns
    -------
    average_word_by_sentence : the average number of words per sentence for body sentences

    """
    nb_words_per_sentence = []
    for part in row["structured_body"][0]["structured_text"]["text"]:
        if part["tags"] == "BODY":
            for sentence in part["part"].split(". "):
                nb_words = len(tokenizer._tokenize(sentence))
                if nb_words > 1:
                    nb_words_per_sentence.append(nb_words)
    average_word_by_sentence = np.round(np.mean(nb_words_per_sentence))
    return average_word_by_sentence
