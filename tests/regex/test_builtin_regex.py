from melusine.regex import (
    EmergencyRegex,
    ReplyRegex,
    ThanksRegex,
    TransferRegex,
    VacationReplyRegex,
)


def test_reply_regex():
    _ = ReplyRegex()


def test_thanks_regex():
    _ = ThanksRegex()


def test_transfer_regex():
    _ = TransferRegex()


def test_vacation_reply_regex():
    regex = VacationReplyRegex()
    regex.test()


def test_emergency_regex():
    regex = EmergencyRegex()
    regex.test()
