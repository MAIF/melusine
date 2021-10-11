import re
from time import time

from melusine import config
from melusine.core.melusine_transformer import MelusineTransformer
from melusine.prepare_email.message import Message


def create_default_transition_patterns():
    transition_patterns = [
        r"De\s*:\s*[^<]*?<?[a-zA-Z0-9._%+-\/=]+\@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}>?\s[;\nAÀ:](?:.{,80}\n){,3}Objet.+\n",
        r"[- ]*?Mail transf[ée]r[ée].*?[;|\n]",
        r"[- ]*?gestionsocietaire@maif.fr a [ée]crit.*?[;|\n]",
        r"Courriel original.+?Objet\s*:.+?[;|\n]",
        r"Transf[ée]r[ée] par.+?Objet\s*:.+?[;|\n]",
        r"Message transmis.+?Objet\s*:.+?[;|\n]",
        r"Message transf[ée]r[ée].+?Objet\s*:.+?[;|\n]",
        r"Message transf[ée]r[ée].+?Pour\s*:.+?[;|\n]",
        r"D[ée]but du message transf[ée]r[ée].+?Objet\s*:.+?[;|\n]",
        r"D[ée]but du message r[ée]exp[ée]di[ée].+?Objet\s*:.+?[;|\n]",
        r"D[ée]but du message transf[ée]r[ée].+?Destinataire\s*:.+?[;|\n]",
        r"mail transf[ée]r[ée].+?Objet\s*:.+?[;|\n]",
        r"Forwarded message.+?To\s*:.+?[;|\n]",
        r"Message d'origine.+?Objet\s*:.+?[;|\n]",
        r"Mail original.+?Objet\s*:.+?[;|\n]",
        r"Original Message.+?Subject\s*:.+?[;|\n]",
        r"Message original.+?Objet\s*:.+?[;|\n]",
        r"Exp[ée]diteur.+?Objet\s*:.+?[;|\n]",
        r"(?:>?[;|\n]?\s*(?:Envoy[ée]|De|Objet|Cc|Envoy[ée] par|Date|A|À|Destinataire|Sent|To|Subject|From|Copie [àa])+?\s*:\s*(?:.*?)\s*[;|\n]\s*)+",
        r"En date de.+?[ée]crit",
        r">?\s*\bLe[^;\n]{0,30}[;|\n]{0,1}[^;\n]{0,30}a[^;\n]{0,30};{0,1}[^;\n]{0,30}[ée]crit\s*:?",
        r">?\s*Message d[eu].+?Objet\s*:.+?[;|\n]",
        r"En date de.+?[ée]crit",
    ]

    return transition_patterns


class EmailSegmenter(MelusineTransformer):
    FILENAME = "email_segmenter.json"

    def __init__(
        self,
        transition_patterns=None,
        input_columns="email",
        output_columns="email",
        strip_characters="\n ;>",
    ):
        super().__init__(
            input_columns=input_columns,
            output_columns=output_columns,
            func=self.segment_email,
        )
        # if not transition_patterns:
        #     transition_patterns = create_default_transition_patterns()
        self.transition_patterns = transition_patterns
        self.strip_characters = strip_characters

        # Compile segmentation regex
        self.compiled_segmentation_regex = self.compile_regex_from_list(
            self.transition_patterns
        )
        self.EXCLUDE_LIST.append("compiled_segmentation_regex")

    def segment_email(self, email):
        text = email.body
        elements = self.segment_text(self.compiled_segmentation_regex, text)

        # Special case : email start with a transition pattern
        if not elements[0].strip(self.strip_characters):
            elements = elements[1:]
        else:
            elements = [""] + elements

        # Create Message instances
        message_list = [
            Message(elements[i + 1], elements[i])
            for i in range(len(elements))
            if i % 2 == 0
        ]

        email.messages = message_list
        return email

    def segment_text(self, compiled_regex, text):

        # Strip start / end characters
        text = text.strip(self.strip_characters)

        # Split text using the compiled segmentation regex
        matches = compiled_regex.split(text)

        return matches

    @staticmethod
    def compile_regex_from_list(regex_list):
        regex_list = ["(?:" + r + ")" for r in regex_list]
        regex = "|".join(regex_list)

        # Add an overall capture group
        regex = "(" + regex + ")"

        return re.compile(regex, re.M)

    def debug_individual_patterns(self, text, skip_fails=False):
        n_match = 0
        for r in self.transition_patterns:
            start = time()
            compiled_regex = self.compile_regex_from_list([r])
            segments = self.segment_text(compiled_regex, text)
            n_segment = len(segments)

            if n_segment == 1:
                if skip_fails:
                    continue
            else:
                n_match += 1

            print(f"\n================================")
            print(f"------- Test regex -------")
            print(r)
            print(f"--------------------------")
            if len(segments) > 1:
                print(f"Number of segments : {len(segments)}")
                for i, m in enumerate(segments):
                    print(f"\nMatch {i} : {m}")
            print(f"Execution time : {time()-start:.2f}")

        print(f"\n================================")
        print(f"Found {n_match} regex matching with the provided text\n")

    def transform(self, df):
        df["email"] = df["email"].apply(self.segment_email)

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
        _: EmailSegmenter
            EmailSegmenter instance
        """
        # Load the object config file
        loaded_dict = cls.load_json(path, filename_prefix=filename_prefix)

        return cls.from_config_or_init(**loaded_dict)
