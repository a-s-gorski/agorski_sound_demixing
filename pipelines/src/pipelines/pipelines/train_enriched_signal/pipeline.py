from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_dataset_enriched,
    test_generative_model,
    train_generative_model,
)


def create_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            node(
                func=compute_dataset_enriched,
                inputs=[
                    "signal_model",
                    "signal_train",
                    "signal_val",
                    "signal_test"],
                outputs=[
                    "signal_train_enriched",
                    "signal_val_enriched",
                    "signal_test_enriched",
                ],
                name="prepare_generative_dataset"),
            node(
                func=train_generative_model,
                inputs=[
                    "signal_train_enriched",
                    "signal_val_enriched",
                    "params:train_waveform_enriched"],
                outputs="signal_enriched_model",
                name="train_generative_model"),
            node(
                func=test_generative_model,
                inputs=[
                    "signal_enriched_model",
                    "signal_test_enriched",
                    "params:train_waveform_enriched",
                    "params:training_waveform",
                    "params:signal_processing",
                ],
                outputs=[
                    "signal_generative_test_metrics",
                    "signal_generative_test_output"],
                name="test_generative_model")])
    return pipeline
