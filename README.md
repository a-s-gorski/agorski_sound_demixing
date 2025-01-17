# Audio source seperation

This project revolves around advancing audio source separation techniques using machine learning. The primary objective is to enhance the outcomes of conventional waveform-based models like HDEMUCS, CONVTASNET, as well as spectrogram-based models such as UNET, through the integration of generative models like GAN and VAE. The project also contains subpackage for training audio source separation models using Cyclic GANs based on: 



The complete project pipeline has been implemented using Kedro and PyTorch Lightning, with experiment tracking facilitated by MLflow.


## Prerequisites

- [Poetry](https://python-poetry.org/docs/#installation)
- Python 3.10
- [Optional] CUDA drivers

## Steps to Get Started

1. Clone the Repository

   ```bash
   git clone https://dagshub.com/a-s-gorski/agorski_sound_demixing.git
   cd agorski_sound_demixing

2. Install hatch

   ```bash
   sudo apt-get install pipx
   pipx install hatch
   ```

## Segmentation Models
3. Download data.
   ```bash
   source scripts/00_



## Preparing data
 Segmentation models.


-- 

## Project Structure

```md
pipelines
├── conf
│   ├── base
│   │   ├── catalog.yml            # Declared datasets
│   │   ├── logging.yml            # Logging configuration
│   │   ├── parameters.yml         # Data processing/training parameters
├── data
│   ├── 01_hq                      # Raw MUSDB18 dataset
│   ├── 02_intermediate            # Torch dataset with preloaded and compressed data
│   ├── 03_signals                 # Processed audio signals
│   ├── 04_spectrograms            # Computed spectrograms
│   ├── 05_signal_dataset          # Input for signal models
│   ├── 05_signal_enriched_dataset # Input for generative models
│   ├── 05_spectrogram_dataset     # Input for frequency domain models
│   ├── 06_models                  # Saved models
│   ├── 07_model_output            # Training results
│   ├── 08_model_score             # Model evaluation results
├── notebooks
│   ├── eda.ipynb                  # Exploratory Data Analysis
├── src
│   ├── pipelines                  # Main project code
│   │   ├── dataset                # Custom Dataset objects
│   │   ├── models                 # PyTorch Lightning-based models
│   │   ├── pipelines              # Kedro nodes/pipelines
│   │   ├── types                  # Enums and Pydantic models for validation
│   │   ├── utils                  # Extracted utility functions
│   ├── tests                      # Unit tests for processing/training
```