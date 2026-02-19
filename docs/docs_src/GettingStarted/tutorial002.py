def run():
    # --8<-- [start:debug_pipeline]
    from melusine.data import load_email_data
    from melusine.pipeline import MelusinePipeline

    # Load an email dataset
    df = load_email_data()

    # Load the default pipeline
    pipeline = MelusinePipeline.from_config("demo_pipeline")

    # Run the pipeline
    df = pipeline.transform(df, debug_mode=True)
    # --8<-- [end:debug_pipeline]
