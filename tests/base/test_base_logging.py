import logging

from melusine.data import load_email_data
from melusine.pipeline import MelusinePipeline


def test_pipeline_debug_logging(caplog):
    with caplog.at_level(logging.DEBUG):
        df = load_email_data()
        pipeline = MelusinePipeline.from_config("demo_pipeline")
        pipeline.transform(df)

        assert "Running transform for Cleaner" in caplog.text
        assert "Running transform for EmergencyDetector" in caplog.text
        assert "Running transform for EmergencyDetector (detect)" in caplog.text
        assert "Running transform for EmergencyDetector (post_detect)" in caplog.text
