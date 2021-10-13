import re
from enum import Enum
from melusine.core.melusine_transformer import MelusineTransformer


class TagScope(Enum):
    last_message = "last_message"
    all = "all"


class EmailTagger(MelusineTransformer):
    FILENAME = "message_tagger.json"

    def __init__(
        self,
        tag_dict=None,
        tag_scope="last_message",
        input_columns="email",
        output_columns="email",
    ):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
        )

        self.tag_dict = tag_dict
        self.tag_scope = TagScope(tag_scope)

    def tag_email(self, email):
        if self.tag_scope is TagScope.last_message:
            messages_to_tag = email.messages[0]
        else:
            messages_to_tag = email.messages

        for m in messages_to_tag:
            self.tag_message(m)

        return email

    def tag_message(self, message):
        text = message.text
        tag_list = list()

        sentences = text.split("\n")
        for sentence in sentences:
            tag = self.tag_text(sentence)
            tag_list.append({"tag": tag, "sentence": sentence})

        message.tag_list = tag_list

    def tag_text(self, text):
        for regex, tag in self.tag_dict.items():
            if re.search(regex, text, re.I):
                print(f"Found match for regex {regex}")
                return tag

        return "BODY"

    def transform(self, df):
        df["email"] = df["email"].apply(self.tag_email)

        return df

    def save(self, path: str, filename_prefix: str = None) -> None:
        """
        Save the Instance into a json file.

        Parameters
        ----------
        path: str
            Save path
        filename_prefix: str
            A prefix to add to the names of the files saved.
        """
        d = self.__dict__.copy()

        # Save Normalizer
        self.save_json(save_dict=d, path=path, filename_prefix=filename_prefix)

    @classmethod
    def load(cls, path: str, filename_prefix: str = None):
        """
        Load the Instance from a json file.

        Parameters
        ----------
        path: str
            Load path
        filename_prefix: str

        Returns
        -------
        _: EmailTagger
            EmailTagger instance
        """
        # Load parameters from json file
        json_data = cls.load_json(path, filename_prefix=filename_prefix)

        return cls.from_json(**json_data)


