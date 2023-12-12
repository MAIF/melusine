"""
The melusine.regex module includes tools for handling regexes.
"""
from melusine.regex.emergency_regex import EmergencyRegex
from melusine.regex.reply_regex import ReplyRegex
from melusine.regex.thanks_regex import ThanksRegex
from melusine.regex.transfer_regex import TransferRegex
from melusine.regex.vacation_reply_regex import VacationReplyRegex

__all__ = ["EmergencyRegex", "ReplyRegex", "ThanksRegex", "TransferRegex", "VacationReplyRegex"]
