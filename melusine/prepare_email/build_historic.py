import re
from melusine.config import ConfigJsonReader

conf_reader = ConfigJsonReader()
config = conf_reader.get_config_file()
regex_transition_list = config['regex']['build_historic']['transition_list']


def build_historic(row):
    """Rebuilds and structures historic of emails from the whole contents.
    Function has to be applied with `apply` method of a DataFrame along an
    axis=1.
    For each email of the historic, it segments the body into 2 different parts
    (2 keys of dict):

    {'text': extract raw text without metadata,
     'meta': get transition from the 'transition_list' defined in the conf.json
     }.


    Parameters
    ----------
    row : row,
        A pandas.DataFrame row object with 'body' column.

    Returns
    -------
    list

    Examples
    --------
        >>> import pandas as pd
        >>> data = pd.read_pickle('./tutorial/data/emails_anonymized.pickle')
        >>> # data contains a 'body' column

        >>> from melusine.prepare_email.build_historic import build_historic
        >>> build_historic(data.iloc[0])  # apply for 1 sample
        >>> data.apply(build_historic, axis=1)  # apply to all samples

    """
    email_body = row['body']
    index_messages, nb_messages = _get_index_transitions(email_body)
    structured_historic = [
        {'text': email_body[index_messages[i][1]:index_messages[i+1][0]],
         'meta': email_body[index_messages[i][0]:index_messages[i][1]]
         } for i in range(nb_messages)]

    return structured_historic


def _get_index_transitions(email_body):
    """Returns list of indexes defining the transitions between
    different messages in an email."""
    index = []
    for regex in regex_transition_list:
        for match in re.finditer(regex, email_body):
            idx = (match.start(), match.end())
            index.append(idx)

    index = [(0, 0)] + index
    index = index + [(len(email_body), len(email_body))]
    index = list(set(index))
    index = sorted(index, key=lambda tup: tup[0])
    index = __filter_overlap(index)
    nb_parts = len(index) - 1

    return index, nb_parts


def __filter_overlap(index):
    """Filters indexes in list if they overlap."""
    if len(index) == 2:
        return index
    index_f = []
    i = 0
    j = i + 1
    while j < len(index):
        if index[i][1] > index[j][0]:
            index[i] = (min(index[i][0], index[j][1]),
                        max(index[i][0], index[j][1]))
            j += 1
        else:
            index_f += [index[i]]
            i = j
            j += 1
    index_f += [index[i]]

    return index_f[:i+1]
