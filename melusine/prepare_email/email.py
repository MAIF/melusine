from melusine.core.melusine_transformer import MelusineTransformer


class Email:
    def __init__(
        self, body, header=None, address_from=None, address_to=None, date=None
    ):
        self.body = body
        self.header = header
        self.address_from = address_from
        self.address_to = address_to
        self.date = date
        self.messages = None

    @property
    def last_message(self):
        if not self.messages:
            raise AttributeError("Message has not been segmented")
        return self.messages[0]

    def segment(self, segmenter):
        self.messages = segmenter.segment_text(self.body)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return repr(self)


class EmailInstanciator(MelusineTransformer):
    FILENAME = "email_instanciator.json"

    def __init__(
        self,
        body_col="body",
        header_col="header",
        from_col="from",
        to_col="to",
        date_col=None,
        input_columns=("body",),
        output_columns=("email",),
    ):
        self.body_col = body_col
        self.header_col = header_col
        self.from_col = from_col
        self.to_col = to_col
        self.date_col = date_col

        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.create_email,
        )

    def create_email(self, row):
        body = row[self.body_col]
        header = row.get(self.header_col)
        address_from = row.get(self.from_col)
        address_to = row.get(self.to_col)
        date = row.get(self.date_col)

        email = Email(
            body=body,
            header=header,
            address_from=address_from,
            address_to=address_to,
            date=date,
        )
        return email

    def transform(self, df):
        df["email"] = df.apply(self.create_email, axis=1)

        return df

    def save(self, path: str, filename_prefix: str = None) -> None:
        pass

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        pass
