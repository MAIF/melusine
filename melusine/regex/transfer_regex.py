from melusine.base import MelusineRegex


class TransferRegex(MelusineRegex):
    """
    Detect transfer patterns in headers such as "tr:".
    """

    @property
    def positive(self) -> str | dict[str, str]:
        """
        Define regex patterns required to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return r"^(tr:|fwd :|tr :|fwd:)"

    @property
    def neutral(self) -> str | dict[str, str] | None:
        """
        Define regex patterns to be ignored when running detection.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    def negative(self) -> str | dict[str, str] | None:
        """
        Define regex patterns prohibited to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return None

    @property
    def match_list(self) -> list[str]:
        """
        List of texts that should activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return [
            "tr: notre discussion",
            "tr : bonjour",
            "Tr : compte rendu",
            "fwd: rdv du 01/01/2001",
            "Fwd: Votre déclaration",
        ]

    @property
    def no_match_list(self) -> list[str]:
        """
        List of texts that should NOT activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return ["transfert d'argent", "re: tr: message du jour" "re:Fwd important notice"]
