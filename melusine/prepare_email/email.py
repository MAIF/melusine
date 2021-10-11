class Email:
    def __init__(self, body, email_from, email_to, date=None):
        self.body = body
        self.email_from = email_from
        self.email_to = email_to
        self.date = date
        self.messages = None

    @property
    def last_message(self):
        if not self.messages:
            raise AttributeError("Message has not been segmented")
        return self.messages[0]

    def segment(self, segmenter):
        self.messages = segmenter.segment_text(self.body)
