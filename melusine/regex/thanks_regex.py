from typing import Dict, List, Optional, Union

from melusine.base import MelusineRegex


class ThanksRegex(MelusineRegex):
    """
    Detect thanks patterns such as "merci".
    """

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        """
        Define regex patterns required to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return r"\bmerci+s?\b|\bremercie?"

    @property
    def neutral(self) -> Optional[Union[str, Dict[str, str]]]:
        """
        Define regex patterns to be ignored when running detection.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    def negative(self) -> Optional[Union[str, Dict[str, str]]]:
        """
        Define regex patterns prohibited to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        forbidden_thanks_words = [
            r"oui",
            r"non",
            r"atten[dt]",
            r"inform[ée]",
            r"proposition",
            r"ci\b",
            r"join[st]?\b",
        ]

        return dict(QUESTION=r"\?", FORBIDDEN_WORDS=r"\b(" + "|".join(forbidden_thanks_words) + ")")

    @property
    def match_list(self) -> List[str]:
        """
        List of texts that should activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return [
            "merci",
            "Je vous remercie pour votre rapidité",
            "un grand MERCI à la MAIF",
            "je tiens à remercier l'équipe",
        ]

    @property
    def no_match_list(self) -> List[str]:
        """
        List of texts that should NOT activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return [
            # réponse à une question ouverte
            "oui, merci à vous",
            "non, merci quand même",
            "merci, j'attends votre réponse",
            "j'aimerais être tenu informée merci",
            "Merci, faites moi une proposition",
            "Merci, ci-joint le formulaire",
            "ci-attaché ledocument, merci",
            "Madame Mercier",
        ]
