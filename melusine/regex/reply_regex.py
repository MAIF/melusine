from melusine.base import MelusineRegex


class ReplyRegex(MelusineRegex):
    """Detect reply patterns in headers such as "re:"."""

    @property
    def positive(self) -> str | dict[str, str]:
        """Define regex patterns required to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.

        """
        return r"^(re:|re :)"

    @property
    def neutral(self) -> str | dict[str, str] | None:
        """Define regex patterns to be ignored when running detection.

        Returns:
            _: Regex pattern or dict of regex patterns.

        """
        return None

    @property
    def negative(self) -> str | dict[str, str] | None:
        """Define regex patterns prohibited to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.

        """
        return None

    @property
    def match_list(self) -> list[str]:
        """List of texts that should activate the MelusineRegex.

        Returns:
            _: List of texts.

        """
        return [
            "re: notre discussion",
            "re : bonjour",
            "Re : compte rendu",
            "RE: rdv du 01/01/2001",
        ]

    @property
    def no_match_list(self) -> list[str]:
        """List of texts that should NOT activate the MelusineRegex.

        Returns:
            _: List of texts.

        """
        return ["renard agile", "tr: re: message du jour" "fwd:re: important notice"]
