from kedro.pipeline import Pipeline, node

from .nodes import (
    inference_spectrogram_node,
    spectrogram_to_signal_node,
    test_signals,
    train_spectrogram_node,
)


def create_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            node(
                func=train_spectrogram_node,
                inputs=[
                    "spectrogram_train",
                    "spectrogram_val",
                    "params:train_spectrogram"],
                outputs="spectrogram_model",
                name="train_spectrogram"),
            node(
                func=inference_spectrogram_node,
                inputs=[
                    "spectrogram_model",
                    "spectrogram_test"],
                outputs="spectrogram_output_frequency",
                name="inference_spectrogram"),
            node(
                func=spectrogram_to_signal_node,
                inputs=[
                    "spectrogram_output_frequency",
                    "params:spectrogram_processing",
                    "original_sizes"],
                outputs="spectrogram_output_waveform",
                name="spectrogram_to_signal"),
            node(
                func=test_signals,
                inputs=[
                    "spectrogram_output_waveform",
                    "params:signal_processing",
                    "params:spectrogram_processing",
                    "params:train_spectrogram"],
                outputs="spectrogram_model_test_metrics",
                name="test_spectrogram",
            )])
    return pipeline
