import re
from melusine import config

regex_transfer_header = config["regex"]["manage_transfer_reply"]["transfer_header"]
regex_answer_header = config["regex"]["manage_transfer_reply"]["answer_header"]
regex_begin_transfer = config["regex"]["manage_transfer_reply"]["begin_transfer"]
regex_begin_transfer_cons = config["regex"]["manage_transfer_reply"][
    "begin_transfer_cons"
]
regex_extract_from = config["regex"]["manage_transfer_reply"]["extract_from"]
regex_extract_to = config["regex"]["manage_transfer_reply"]["extract_to"]
regex_extract_date = config["regex"]["manage_transfer_reply"]["extract_date"]
regex_extract_header = config["regex"]["manage_transfer_reply"]["extract_header"]


def add_boolean_transfer(row):
    """Compute boolean Series which return True if the "header" starts with given
    regex 'answer_subject', False if not.

    To be used with methods such as: `apply(func, axis=1)` or
    `apply_by_multiprocessing(func, axis=1, **kwargs)`.

    Parameters
    ----------
    row : row of pd.Dataframe, columns ['header']

    Returns
    -------
    pd.Series

    Examples
    --------
        >>> import pandas as pd
        >>> data = pd.read_pickle('./tutorial/data/emails_anonymized.pickle')
        >>> # data contains a 'header' column

        >>> from melusine.prepare_email.manage_transfer_reply import add_boolean_transfer
        >>> add_boolean_transfer(data.iloc[0])  # apply for 1 sample
        >>> data.apply(add_boolean_transfer, axis=1)  # apply to all samples

    """
    is_transfer = False
    try:
        if re.match(regex_transfer_header, row["header"]):
            is_transfer = True
    except Exception:
        pass

    return is_transfer


def add_boolean_answer(row):
    """Compute boolean Series which return True if the "header" starts with given
    regex 'transfer_subject', False if not.

    To be used with methods such as: `apply(func, axis=1)` or
    `apply_by_multiprocessing(func, axis=1, **kwargs)`.

    Parameters
    ----------
    row : row of pd.Dataframe, columns ['header']

    Returns
    -------
    pd.Series

    Examples
    --------
        >>> import pandas as pd
        >>> data = pd.read_pickle('./tutorial/data/emails_anonymized.pickle')
        >>> # data contains a 'header' column

        >>> from melusine.prepare_email.manage_transfer_reply import add_boolean_answer
        >>> add_boolean_answer(data.iloc[0])  # apply for 1 sample
        >>> data.apply(add_boolean_answer, axis=1)  # apply to all samples

    """
    is_answer = False
    try:
        if re.match(regex_answer_header, row["header"]):
            is_answer = True
    except Exception:
        pass

    return is_answer


def check_mail_begin_by_transfer(row):
    """Compute boolean Series which return True if the "body" starts with given
    regex 'begin_transfer', False if not.

    To be used with methods such as: `apply(func, axis=1)` or
    `apply_by_multiprocessing(func, axis=1, **kwargs)`.

    Parameters
    ----------
    row : row of pd.Dataframe, columns ['body']

    Returns
    -------
    pd.Series

    Examples
    --------
        >>> import pandas as pd
        >>> data = pd.read_pickle('./tutorial/data/emails_anonymized.pickle')
        >>> # data contains a 'body' column

        >>> from melusine.prepare_email.manage_transfer_reply import check_mail_begin_by_transfer
        >>> check_mail_begin_by_transfer(data.iloc[0])  # apply for 1 sample
        >>> data.apply(check_mail_begin_by_transfer, axis=1)  # apply to all samples

    """
    is_begin_by_transfer = False
    try:
        if re.search(regex_begin_transfer, row["body"]):
            is_begin_by_transfer = True
        if re.search(regex_begin_transfer_cons, row["body"]):
            is_begin_by_transfer = True
    except Exception:
        pass

    return is_begin_by_transfer


def update_info_for_transfer_mail(row):
    """Extracts and updates informations from forwarded mails, such as: body,
    from, to, header, date.
    - It changes the header by the initial subject (extracted from forward
    email).
    - It removes the header from emails' body.

    To be used with methods such as: `apply(func, axis=1)` or
    `apply_by_multiprocessing(func, axis=1, **kwargs)`.

    Parameters
    ----------
    row : row of pd.Dataframe,
    columns ['body', 'header', 'from', 'to', 'date', 'is_begin_by_transfer']

    Returns
    -------
    pd.DataFrame

    Examples
    --------
        >>> import pandas as pd
        >>> from melusine.prepare_email.manage_transfer_reply import check_mail_begin_by_transfer
        >>> data = pd.read_pickle('./tutorial/data/emails_anonymized.pickle')
        >>> data['is_begin_by_transfer'] = data.apply(check_mail_begin_by_transfer, axis=1)
        >>> # data contains columns ['from', 'to', 'date', 'header', 'body', 'is_begin_by_transfer']

        >>> from melusine.prepare_email.manage_transfer_reply import update_info_for_transfer_mail
        >>> update_info_for_transfer_mail(data.iloc[0])  # apply for 1 sample
        >>> data.apply(update_info_for_transfer_mail, axis=1)  # apply to all samples

    """
    try:
        if row["is_begin_by_transfer"]:
            row["from"] = re.split(regex_extract_from, row["body"])[1]
            row["to"] = re.split(regex_extract_to, row["body"])[1]
            row["date"] = re.split(regex_extract_date, row["body"])[1]
            row["header"] = re.split(regex_extract_header, row["body"])[1]
            row["body"] = "".join(
                row["body"].split(re.findall(regex_extract_header, row["body"])[0])[1:]
            )

    except Exception:
        pass

    return row
