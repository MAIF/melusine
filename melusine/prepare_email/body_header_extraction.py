def extract_last_body(row):
    """Extracts the body from the last message of the conversation.
    The conversation is structured as a dictionary.

    To be used with methods such as: `apply(func, axis=1)` or
    `apply_by_multiprocessing(func, axis=1, **kwargs)`.

    Parameters
    ----------
    message_dict : dict

    Returns
    -------
    str

    """
    last_message_dict = row["structured_body"][0]
    last_body = extract_body(last_message_dict)

    return last_body


def extract_body(message_dict):
    """Extracts the body from a message dictionary.

    Parameters
    ----------
    message_dict : dict

    Returns
    -------
    str

    """
    tagged_parts_list = message_dict["structured_text"]["text"]
    body = ""
    for part_tag_dict in tagged_parts_list:
        part = part_tag_dict["part"]
        tag = part_tag_dict["tags"]
        if tag == "BODY":
            body += part + " "
        elif tag == "GREETINGS":
            break

    return body


def extract_header(message_dict):
    """Extracts the header from a message dictionary.

    Parameters
    ----------
    message_dict : dict

    Returns
    -------
    str

    """
    header = message_dict["structured_text"]["header"]

    return header
