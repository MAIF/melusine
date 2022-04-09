import pytest
from melusine.nlp_tools.emoji_flagger import DeterministicEmojiFlagger


@pytest.mark.parametrize(
    "input_text, output_text",
    [
        ("plusieurs fois 😄", "plusieurs fois  flag_emoji_ "),
        ("comme un grand :)", "comme un grand :)"),
    ],
)
def test_emoji_flagger(input_text, output_text):
    emoji_flagger = DeterministicEmojiFlagger()

    flagged_text = emoji_flagger._flag_emojis(input_text)

    assert flagged_text == output_text