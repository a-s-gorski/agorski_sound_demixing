from kedro.pipeline import Pipeline, node

from .nodes import test_model_node, train_model_node


def create_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            node(
                func=train_model_node,
                inputs=[
                    "signal_train",
                    "signal_val",
                    "params:training_waveform",
                    "params:random_state"],
                outputs="signal_model",
                name="train_waveform"),
            node(
                func=test_model_node,
                inputs=[
                    "signal_model",
                    "signal_test",
                    "params:signal_processing",
                    "params:training_waveform"],
                outputs=[
                    "signal_model_test_metrics",
                    "signal_model_test_output"],
                name="test_waveform")])
    return pipeline
