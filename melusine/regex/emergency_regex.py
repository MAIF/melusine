from typing import Dict, List, Optional, Union

from melusine.base import MelusineRegex


class EmergencyRegex(MelusineRegex):
    """
    Detect reply patterns in headers such as "re:".
    """

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        """
        Define regex patterns required to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return r"urgent|emergency"

    @property
    def neutral(self) -> Optional[Union[str, Dict[str, str]]]:
        """
        Define regex patterns to be ignored when running detection.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return r"emergency exit"

    @property
    def negative(self) -> Optional[Union[str, Dict[str, str]]]:
        """
        Define regex patterns prohibited to activate the MelusineRegex.

        Returns:
            _: Regex pattern or dict of regex patterns.
        """
        return dict(
            blacklist=r"Mrs. TooInsistent|Mr. Annoying",
            not_my_business=r"GalaxyFarFarAway",
        )

    @property
    def match_list(self) -> List[str]:
        """
        List of texts that should activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return [
            "We have an emergency",
            "This message is urgent",
        ]

    @property
    def no_match_list(self) -> List[str]:
        """
        List of texts that should NOT activate the MelusineRegex.

        Returns:
            _: List of texts.
        """
        return [
            "Mr. Annoying is calling for an emergency",
            "Mrs. TooInsistent called 8 times for an urgent matter",
            "There is an emergency in GalaxyFarFarAway",
        ]
