def print_color_mail(structured_body):
    """Highlight the tagged sentences.

    Parameters
    ----------
    structured_body : a structured body from process_sent_tag,

    Returns
    -------
    Print the mail by sentence.

    """
    for message in structured_body:
        print("___________________________\n")
        print_color(str(message.get("meta")), "META")

        structured_text = message.get("structured_text")
        header = structured_text.get("header")
        print_color(str(header), "HEADER")

        for sentence in structured_text.get("text"):
            text = sentence.get("part")
            tag = sentence.get("tags")
            print_color(text, tag)


def print_color(text, part=None):
    """Select according to the tag the right color to use when printing."""
    text = text.replace("\r", "")
    switcher_tag = {
        "HELLO": "\033[0;37;44m" + "HELLO" + "\033[0m",
        "GREETINGS": "\033[0;37;45m" + "GREETINGS" + "\033[0m",
        "SIGN": "\033[0;37;41m" + "SIGN" + "\033[0m",
        "THANKS": "\033[0;31;46m" + "THANKS" + "\033[0m",
        "PJ": "\033[0;37;42m" + "PJ" + "\033[0m",
        "META": "\33[43m" + "META" + "\033[0m",
        "FOOTER": "\33[41m" + "FOOTER" + "\033[0m",
        "DISCLAIMER": "\33[41m" + "DISCLAIMER" + "\033[0m",
        "TYPO": "\33[47m" + "TYPO" + "\033[0m",
        "HEADER": "\033[0;37;41m" + "HEADER" + "\033[0m",
    }

    switcher = {
        "HELLO": "\033[0;37;44m" + text + "\033[0m",
        "GREETINGS": "\033[0;37;45m" + text + "\033[0m",
        "SIGN": "\033[0;37;41m" + text + "\033[0m",
        "THANKS": "\033[0;31;46m" + text + "\033[0m",
        "PJ": "\033[0;37;42m" + text + "\033[0m",
        "META": "\33[43m" + text + "\033[0m",
        "FOOTER": "\33[41m" + text + "\033[0m",
        "DISCLAIMER": "\33[41m" + text + "\033[0m",
        "TYPO": "\33[47m" + text + "\033[0m",
        "HEADER": "\033[0;37;41m" + text + "\033[0m",
    }

    if part in switcher.keys():
        print(
            "> ",
            switcher_tag.get(part, text),
            " : ",
            switcher.get(part, text),
            "\n",
        )
    else:
        print("> BODY : ", switcher.get(part, text), "\n")
