from melusine.regex import (
    EmergencyRegex,
    ReplyRegex,
    ThanksRegex,
    TransferRegex,
    VacationReplyRegex,
)


def test_reply_regex():
    regex = ReplyRegex()
    regex.test()


def test_thanks_regex():
    regex = ThanksRegex()
    regex.test()


def test_transfer_regex():
    regex = TransferRegex()
    regex.test()


def test_vacation_reply_regex():
    regex = VacationReplyRegex()
    regex.test()


def test_emergency_regex():
    regex = EmergencyRegex()
    regex.test()
