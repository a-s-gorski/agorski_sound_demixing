from kedro.pipeline import Pipeline, node

from .nodes import process_dataset_signals


def create_pipeline():
    pipeline = Pipeline([
        node(
            func=process_dataset_signals,
            inputs=["params:data_processing_paths", "params:data_processing"],
            outputs="base_dataset",
            name="read_signals",
        ),
    ])

    return pipeline
