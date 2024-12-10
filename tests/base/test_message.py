import re

from melusine import config
from melusine.message import Message


def test_message_repr():
    message = Message(text="Hello")

    assert re.search(r"meta='NA'", repr(message))
    assert re.search(r"text='Hello'", repr(message))

    message = Message(text="Hello", meta="someone@domain.fr")

    assert re.search(r"meta='someone@domain.fr'", repr(message))
    assert re.search(r"text='Hello'", repr(message))


def test_message_has_tags():
    message = Message(text="Hello")
    message.tags = [
        {"base_text": "Bonjour", "base_tag": "HELLO"},
        {"base_text": "Pouvez-vous", "base_tag": "BODY"},
        {"base_text": "Cordialement", "base_tag": "GREETINGS"},
    ]

    assert not message.has_tags(target_tags=["FOOTER"])
    assert message.has_tags(target_tags=["BODY"])
    assert message.has_tags(target_tags=["FOOTER", "HELLO"])


def test_message_has_tags_stop_at():
    message = Message(text="Hello")
    message.tags = [
        {"base_text": "Bonjour", "base_tag": "HELLO"},
        {"base_text": "Cordialement", "base_tag": "GREETINGS"},
        {"base_text": "Blah Blah Blah", "base_tag": "BODY"},
    ]

    assert not message.has_tags(target_tags=["BODY"], stop_at=["GREETINGS"])


def test_message_has_tags_no_tags():
    message = Message(text="Hello")

    assert not message.has_tags(target_tags=["BODY"])


def test_message_extract_parts():
    message = Message(text="Hello")
    message.tags = [
        {"base_text": "Bonjour", "base_tag": "HELLO"},
        {"base_text": "Pouvez-vous", "base_tag": "BODY"},
        {"base_text": "Cordialement", "base_tag": "GREETINGS"},
    ]

    assert message.extract_parts(target_tags={"BODY"}) == [{"base_text": "Pouvez-vous", "base_tag": "BODY"}]
    assert message.extract_parts(target_tags=["GREETINGS", "HELLO"]) == [
        {"base_text": "Bonjour", "base_tag": "HELLO"},
        {"base_text": "Cordialement", "base_tag": "GREETINGS"},
    ]


def test_message_extract_parts_stop():
    message = Message(text="Hello")
    message.tags = [
        {"base_text": "Bonjour", "base_tag": "HELLO"},
        {"base_text": "Envoyé depuis mon Iphone", "base_tag": "FOOTER"},
        {"base_text": "Cordialement", "base_tag": "GREETINGS"},
        {"base_text": "Blah Blah Blah", "base_tag": "BODY"},
    ]

    extracted = message.extract_parts(target_tags=["BODY"], stop_at=["FOOTER", "GREETINGS"])

    assert extracted == []


def test_message_extract_parts_no_tags():
    message = Message(text="Hello")

    assert not message.extract_parts(target_tags={"BODY"})


def test_message_extract_last_body():
    message = Message(text="Hello")
    message.tags = [
        {"base_text": "Bonjour", "base_tag": "HELLO"},
        {"base_text": "Pouvez-vous", "base_tag": "BODY"},
        {"base_text": "Cordialement", "base_tag": "GREETINGS"},
    ]

    assert message.extract_last_body() == [{"base_text": "Pouvez-vous", "base_tag": "BODY"}]


def test_str():
    # Arrange
    message = Message(meta="Test\nmeta", text="Hello")
    message.tags = [
        {"base_text": "ABC", "base_tag": "TAG"},
        {"base_text": "ABCD", "base_tag": "TAAG"},
        {"base_text": "ABCDE", "base_tag": "TAAAG"},
    ]

    expected_list = [
        r"=+ +Message +=+",
        r"-+ +Meta +-+",
        r"Test",
        r"meta",
        r"-+ +Text +-+",
        r"ABC\.+TAG",
        r"ABCD\.+TAAG",
        r"ABCDE\.+TAAAG",
        r"=+",
    ]

    # Act
    result = str(message).strip()

    # Assert
    assert len(result.splitlines()) == len(expected_list)
    for text_line, regex in zip(result.splitlines(), expected_list):
        assert re.match(regex, text_line)


def test_str_no_meta():
    # Arrange
    message = Message(text="Hello")
    message.tags = [
        {"base_text": "ABC", "base_tag": "TAG"},
        {"base_text": "ABCD", "base_tag": "TAAG"},
        {"base_text": "ABCDE", "base_tag": "TAAAG"},
    ]

    expected_list = [
        r"=+ +Message +=+",
        r"-+ +Meta +-+",
        r"N/A",
        r"-+ +Text +-+",
        r"ABC\.+TAG",
        r"ABCD\.+TAAG",
        r"ABCDE\.+TAAAG",
        r"=+",
    ]

    # Act
    result = str(message).strip()

    # Assert
    assert len(result.splitlines()) == len(expected_list)
    for text_line, regex in zip(result.splitlines(), expected_list):
        assert re.match(regex, text_line)


def test_str_no_tags():
    # Arrange
    message = Message(text="Hello")

    expected_list = [
        r"=+ +Message +=+",
        r"-+ +Meta +-+",
        r"N/A",
        r"-+ +Text +-+",
        r"Hello",
        r"=+",
    ]

    # Act
    result = str(message).strip()

    # Assert
    assert len(result.splitlines()) == len(expected_list)
    for text_line, regex in zip(result.splitlines(), expected_list):
        assert re.match(regex, text_line)


def test_str_no_conf(reset_melusine_config):
    config.reset({"Test": "Test"})
    message = Message(text="test", tags=[{"base_text": "TEST TEXT", "base_tag": "TEST TAG"}])
    print(message)
    assert message.__str__()
