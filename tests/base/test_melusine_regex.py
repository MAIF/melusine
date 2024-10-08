from typing import Any, Dict, List, Optional, Union

import pytest

from melusine.base import MelusineRegex


class VirusRegex(MelusineRegex):
    """
    Detect computer viruses but not software bugs.
    """

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        return r"virus"

    @property
    def neutral(self) -> Optional[Union[str, Dict[str, str]]]:
        return dict(
            NEUTRAL_MEDICAL_VIRUS="corona virus",
            NEUTRAL_INSECT="ladybug",
        )

    @property
    def negative(self) -> Optional[Union[str, Dict[str, str]]]:
        return dict(
            NEGATIVE_BUG="bug",
        )

    @property
    def match_list(self) -> List[str]:
        return [
            "This email contains a virus",
            "There is a virus in the ladybug software",
            "The corona virus is not a computer virus",
        ]

    @property
    def no_match_list(self) -> List[str]:
        return [
            "This process just had a bug",
            "This is a bug not a virus",
            "There are ladybugs on the windows",
        ]


def test_erroneous_substitution_pattern():
    with pytest.raises(ValueError):
        _ = VirusRegex(substitution_pattern="12345")


def test_method_test():
    regex = VirusRegex()
    regex.test()


def test_match_method():
    regex = VirusRegex()
    match_data = regex("The computer virus in the ladybug software caused a bug in the corona virus dashboard")

    assert match_data[MelusineRegex.MATCH_RESULT] is False
    assert match_data[MelusineRegex.POSITIVE_MATCH_FIELD] == {
        "DEFAULT": [{"match_text": "virus", "start": 13, "stop": 18}]
    }
    assert match_data[MelusineRegex.NEUTRAL_MATCH_FIELD] == {
        "NEUTRAL_INSECT": [{"match_text": "ladybug", "start": 26, "stop": 33}],
        "NEUTRAL_MEDICAL_VIRUS": [{"match_text": "corona virus", "start": 63, "stop": 75}],
    }
    assert match_data[MelusineRegex.NEGATIVE_MATCH_FIELD] == {
        "NEGATIVE_BUG": [{"match_text": "bug", "start": 52, "stop": 55}]
    }


def test_direct_match_method():
    regex = VirusRegex()

    bool_match_result = regex.get_match_result("The computer virus")

    assert bool_match_result is True

    bool_match_result = regex.get_match_result(
        "The computer virus in the ladybug software caused a bug in the corona virus dashboard"
    )

    assert bool_match_result is False


def test_describe_method(capfd):
    """
    Test describe method.
    """
    regex = VirusRegex()

    # Negative match on bug (group NEGATIVE_BUG) and ignore ladybug and corona virus
    regex.describe("The computer virus in the ladybug software caused a bug in the corona virus dashboard")
    out, _ = capfd.readouterr()
    assert "NEGATIVE_BUG" in out
    assert "start" not in out

    # Same but include match positions
    regex.describe(
        "The computer virus in the ladybug software caused a bug in the corona virus dashboard",
        position=True,
    )
    out, _ = capfd.readouterr()
    assert "match result is : NEGATIVE" in out
    assert "NEGATIVE_BUG" in out
    assert "start" in out

    regex.describe("This is a dangerous virus")
    out, _ = capfd.readouterr()
    assert "match result is : POSITIVE" in out
    assert "start" not in out

    regex.describe("Nada")
    out, _ = capfd.readouterr()
    assert "The input text did not match anything" in out


def test_repr():
    """
    Test __repr__ method
    """
    regex = VirusRegex()
    assert "VirusRegex" in repr(regex)


def test_default_neutral_and_negative():
    """
    Test a regex class using default neutral and negative properties.
    """

    class SomeRegex(MelusineRegex):
        """
        Test class.
        """

        @property
        def positive(self):
            return r"test"

        @property
        def match_list(self):
            return ["test"]

        @property
        def no_match_list(self):
            return ["bip bip"]

    regex = SomeRegex()
    assert regex.neutral is None
    assert regex.negative is None


class PreMatchHookVirusRegex(VirusRegex):
    def pre_match_hook(self, text: str) -> str:
        text = text.replace("virrrrus", "virus")
        return text


def test_pre_match_hook():
    reg = PreMatchHookVirusRegex()

    bool_match_result = reg.get_match_result("I see a virrrrus !")

    assert bool_match_result is True


class PostMatchHookVirusRegex(VirusRegex):
    def post_match_hook(self, match_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Test custom post processing of match data"""
        if (
            match_dict[self.MATCH_RESULT] is True
            and "NEUTRAL_MEDICAL_VIRUS" in match_dict[self.NEUTRAL_MATCH_FIELD]
            and "NEUTRAL_INSECT" in match_dict[self.NEUTRAL_MATCH_FIELD]
        ):
            match_dict[self.MATCH_RESULT] = False

        return match_dict


def test_post_match_hook():
    reg = PostMatchHookVirusRegex()

    bool_match_result = reg.get_match_result("I see a virus, a corona virus and a ladybug")
    assert bool_match_result is False

    bool_match_result = reg.get_match_result("I see a virus and a ladybug")
    assert bool_match_result is True


class PairedMatchRegex(MelusineRegex):
    """
    Test paired matching.
    """

    @property
    def positive(self) -> Union[str, Dict[str, str]]:
        return {
            "test_1": r"pos_pattern_1",
            "test_2": r"pos_pattern_2",
        }

    @property
    def negative(self) -> Optional[Union[str, Dict[str, str]]]:
        return {
            "_test_1": r"neg_pattern_1",
            "generic": r"neg_pattern_2",
        }

    @property
    def match_list(self) -> List[str]:
        return [
            "Test pos_pattern_1",
            "pos_pattern_2",
            "pos_pattern_2 and neg_pattern_1",
        ]

    @property
    def no_match_list(self) -> List[str]:
        return [
            "test",
            "Test pos_pattern_1 and neg_pattern_1",
            "pos_pattern_2 and neg_pattern_2",
            "pos_pattern_1 and neg_pattern_2",
        ]


def test_paired_matching_test():
    regex = PairedMatchRegex()
    regex.test()
