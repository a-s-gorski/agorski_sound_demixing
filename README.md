# Audio source seperation

This project revolves around advancing audio source separation techniques using machine learning. The primary objective is to enhance the outcomes of conventional waveform-based models like HDEMUCS, CONVTASNET, as well as spectrogram-based models such as UNET, through the integration of generative models like GAN and VAE.

The complete project pipeline has been implemented using Kedro and PyTorch Lightning, with experiment tracking facilitated by MLflow. The project structure adheres to Kedro's conventions.

## Prerequisites

- [Poetry](https://python-poetry.org/docs/#installation)
- Python 3.10
- [Optional] CUDA drivers

## Steps

1. Clone the Repository and download data

   ```bash
   git clone https://dagshub.com/a-s-gorski/agorski_sound_demixing.git
   cd agorski_sound_demixing
   sudo apt-get install dvc
   dvc pull -r

2. Install packages and run poetry

   ```bash
   poetry install
   poetry shell

3. Run kedro

   ```bash
   cd pipelines
   kedro run

```md
pipelines
├── conf
│   ├── base
│   │   ├── catalog.yml - Declared datasets
│   │   ├── logging.yml - Logging configuration
│   │   ├── parameters.yml - Data processing / training parameters
├── data 
│   ├── 01_hq - Raw MusDB18 dataset
│   ├── 02_intermediate - Torch dataset with preloaded and compressed data
│   ├── 03_signals - Processed audio signals
│   ├── 04_spectrograms - Computed spectrograms
│   ├── 05_signal_dataset - Input for signal models
│   ├── 05_signal_enriched_dataset - Input for generative models
│   ├── 05_spectrogram_dataset - Input for frequency domain models
│   ├── 06_models - Saved models
│   ├── 07_model_output - Training results
│   ├── 08_model_score - Model results on the test set
├── notebooks 
│   ├── eda.ipynb - Notebook for exploratory data analysis
├── src - Implementation
│   ├── pipelines - Main package
│   │   ├── dataset - Custom torch-based Dataset objects
│   │   ├── models - PyTorch Lightning based models
│   │   ├── pipelines - Kedro nodes/pipelines
│   │   ├── types - Enums / Pydantic models for configuration / validation
│   │   ├── utils - Extracted utility functions
│   ├── tests - Testing processing / training functions
├── sdx-2023-music-demixing-track-starter-kit - Directory for submitting to the 2023 audio source separation challenge
```