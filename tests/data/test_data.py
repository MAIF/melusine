from melusine.data import load_email_data


def test_load_data():
    df = load_email_data()
    assert "body" in df
