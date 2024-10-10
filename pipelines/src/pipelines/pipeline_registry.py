"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from pipelines.pipelines.dataset_reading import pipeline as data_processing_pipeline
from pipelines.pipelines.prepare_training_signal import (
    pipeline as prepare_training_signal_pipeline,
)
from pipelines.pipelines.prepare_training_spectrogram import (
    pipeline as prepare_training_spectrogram_pipeline,
)
from pipelines.pipelines.train_enriched_signal import (
    pipeline as train_enriched_signal_pipeline,
)
from pipelines.pipelines.train_spectrogram import pipeline as train_spectrogram_pipeline
from pipelines.pipelines.train_waveform import pipeline as train_waveform_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["data_processing"] = data_processing_pipeline.create_pipeline()
    pipelines["preprocess_signal"] = prepare_training_signal_pipeline.create_pipeline()
    pipelines["preprocess_spectrogram"] = prepare_training_spectrogram_pipeline.create_pipeline()
    pipelines["train_waveform"] = train_waveform_pipeline.create_pipeline()
    pipelines["train_spectrogram"] = train_spectrogram_pipeline.create_pipeline()
    pipelines["train_waveform_generative"] = train_enriched_signal_pipeline.create_pipeline()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
