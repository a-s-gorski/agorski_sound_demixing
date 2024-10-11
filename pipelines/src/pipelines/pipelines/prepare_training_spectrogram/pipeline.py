from kedro.pipeline import Pipeline, node

from .nodes import preproces_spectrogram_node, spectrogram_split_node


def create_pipeline():
    pipeline = Pipeline([
        node(
            func=preproces_spectrogram_node,
            inputs=["processed_signal_dataset", "params:spectrogram_processing"],
            outputs=["processed_spectrogram_dataset", "original_sizes"],
            name="preprocess_spectogram"
        ),
        node(
            func=spectrogram_split_node,
            inputs=[
                "processed_spectrogram_dataset",
                "params:test_size",
                "params:random_state"],
            outputs=["spectrogram_train", "spectrogram_val", "spectrogram_test"],
            name="split_spectrogram"
        )
    ])
    return pipeline
