def run():
    # --8<-- [start:simple_pipeline]
    from melusine.data import load_email_data
    from melusine.pipeline import MelusinePipeline

    # Load an email dataset
    df = load_email_data()

    # Load a pipeline
    pipeline = MelusinePipeline.from_config("demo_pipeline")  # (1)!

    # Run the pipeline
    df = pipeline.transform(df)
    # --8<-- [end:simple_pipeline]
