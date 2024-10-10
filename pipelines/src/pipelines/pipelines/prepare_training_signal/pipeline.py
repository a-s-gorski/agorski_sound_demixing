from kedro.pipeline import Pipeline, node

from .nodes import signal_preprocessing_node, signal_split_node


def create_pipeline():
    pipeline = Pipeline([
        node(
            func=signal_preprocessing_node,
            inputs=["base_dataset", "params:signal_processing"],
            outputs="processed_signal_dataset",
            name="preprocess_signal",
        ),
        node(
            func=signal_split_node,
            inputs=["processed_signal_dataset", "params:test_size", "params:random_state"],
            outputs=["signal_train", "signal_val", "signal_test"],
            name="split_signal"
        )
    ])
    return pipeline
