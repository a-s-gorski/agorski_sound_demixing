# Audio source seperation
This repository represents experiments with various types of architectures (segmentation models, GANs and diffusion models) for the purpose of improving
music source separation on musdb18 dataset. It provide installable python package and scripts related to data downloading, pre-processing, model training,
evaluation and inference.


## Prerequisites

- Uv - https://docs.astral.sh/uv/getting-started/installation/
- Python 3.8
- [Optional] CUDA drivers
- Make and g++ compiler installed.

## Steps to Get Started

1. Clone the Repository

   ```bash
   git clone https://dagshub.com/a-s-gorski/agorski_sound_demixing.git
   cd agorski_sound_demixing

2. Install uv

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install the package and the dependencies

   ```bash
   uv pip install -e .
   ```

4. Segmentation models need additonal dependencies:
   ```bash
   source ./scripts/03_install_inplace_abn.sh
   ```


Optionally you can use scripts to install conda and the package with cuda.
   ```bash
   source ./scripts/01_setup_environnment.sh
   source ./scripts/02_install_gpu.sh
   source ./scripts/03_install_inplace_abn.sh
   source ./scripts/04_build_install_package.sh
   ```


# Segmentation Models

Segmentation models are neural network architectures designed to separate different sound sources from a mixed audio signal. These models process audio spectrograms and learn to extract individual components such as vocals, accompaniment, bass, drums, and other instruments. This repository includes several segmentation-based models, including ResUNet, U-Net, and MobileNet, which are trained to improve source separation quality.

The available models can be used to separate vocals from accompaniment, individual instrument stems, or multitrack separation depending on the training setup. Below are the steps to prepare data and train the models.


1. Download data
```bash
source scripts/scripts/00_download_data.sh
```
2. Preprocess data
```bash
python3.8 scripts/05_convert_to_hdf5.py
python3.8 scripts/06_prepare_indexes.py
source scripts/07_prepare_evaluation_audios.sh
```
3. Run training
The easiest way to start training is to use the predefined Makefile commands. For example:

```
make train_vocals_accompaniment_resunet_ismir2021
```

The full list of available training commands:

- train_accompaniment_vocals_mobilenet_subbandtime - Train a MobileNet-based model using sub-band time processing.
- train_accompaniment_vocals_resunet_ismir2021 - Train a ResUNet model following ISMIR 2021 specifications.
- train_accompaniment_vocals_resunet_subbandtime - Train a ResUNet model with sub-band time separation.
- train_accompaniment_vocals_resunet - Train a standard ResUNet model for accompaniment and vocals separation.
- train_accompaniment_vocals_unet - Train a U-Net model for accompaniment and vocals separation.
- train_vocals_accompaniment_mobilenet_subbandtime - Train a MobileNet-based model for vocal and accompaniment separation using sub-band time.
- train_vocals_accompaniment_resunet_ismir2021 - Train a ResUNet model for vocal and accompaniment separation following ISMIR 2021.
- train_vocals_accompaniment_resunet_subbandtime - Train a ResUNet model for vocal and accompaniment separation using sub-band time.
- train_vocals_accompaniment_resunet - Train a standard ResUNet model for vocal and accompaniment separation.
- train_vocals_accompaniment_unet - Train a U-Net model for vocal and accompaniment separation.
- train_vocals_bass_drums_other_resunet_subbandtime - Train a ResUNet model for full multitrack separation (vocals, bass, drums, other) using sub-band time.
- train_vocals_bass_drums_other_unet - Train a U-Net model for full multitrack separation.

You can modify each training configuration by editing its corresponding YAML file in the configs/training/ directory.

For example, to adjust the training parameters for vocal separation with ResUNet ISMIR2021, edit the following file: configs/training/vocals-accompaniment,resunet_ismir2021.yaml.

Inside this file, you can change various training parameters such as:

- Learning rate
- Batch size
- Number of epochs
- Model architecture settings
- Dataset preprocessing options

Make sure to update the configuration before running training commands to apply your custom settings

4. Run Separation for a Single File
```bash
make separate_file
```
- Update the Makefile to modify input/output paths and the path to the checkpoint.
- Set the TRAIN_CONFIG environment variable to specify the configuration.

5. To run separation for a full directory you might want to run:
```bash
make separate_dir
```
- Update the Makefile to modify input/output paths and the path to the checkpoint.'
- Set the TRAIN_CONFIG environment variable to specify the configuration.
# Generative Adversarial Networks (GANs)

Generative Adversarial Networks (GANs) are a class of machine learning models designed to generate realistic synthetic data. They consist of two competing neural networks: a **generator**, which creates new data samples, and a **discriminator**, which evaluates their authenticity. Through an adversarial process, the generator improves over time, creating increasingly realistic outputs.

This project utilizes a specific modification of GANs known as **CycleGAN**, which is particularly useful for tasks where paired training data is unavailable. The implementation of CycleGAN in this project is adapted from [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Data Preparation
GANs require a high-quality version of the **MUSDB18 dataset** for training. If you haven't downloaded it yet, execute the following script:

```bash
source scripts/gan/00_download_data.sh
```

### Preprocessing Data
Before training, the dataset needs to be preprocessed:

```bash
python scripts/gan/04_prepare_data.py
```

### Training the Model
Once the dataset is prepared, you can start training:

```bash
python scripts/gan/05_train.py
```

### Configurations
You can modify the training configuration by editing:

```
configs/gan/config.yaml
```

---

# Diffusion Models

Diffusion models are a class of generative models that learn to create data samples by iteratively denoising a randomly sampled noise vector. They have recently shown great promise in various tasks, including **audio source separation**. The diffusion model in this project is based on the implementation from [Gladia Research Group](https://github.com/gladia-research-group/multi-source-diffusion-models) and utilizes the `audio_diffusion_pytorch` library.

## Data Preparation
The diffusion model requires the same **MUSDB18 dataset** as the GANs. If you have already downloaded it, you can skip this step. Otherwise, run:

```bash
source scripts/diffusion_models/00_download_data.sh
```

### Preprocessing Data
To preprocess the dataset for training:

```bash
python scripts/diffusion_models/01_prepare_data.py
```

### Training the Model
To train the diffusion model, use the following command:

```bash
make diffusion_train_model
```

### Configurations
To modify training parameters, you can edit the following configuration files:

- **Dataset configuration:** `configs/diffusion_model/dataset.yaml`
- **Model configuration:** `configs/diffusion_model/training.yaml`


## About the package (for developers).

### Uv

The primary dependency and build management tool for this package is `uv`. It can be installed as mentioned in previous examples, although it is only recommended, not required. You can still install dependencies using:

```sh
pip install -e .
```
without uv, but the installation process will take significantly longer. After experimenting with multiple package managers, I found uv to be significantly faster than Hatch or Poetry.
It also supports .lock files and custom sources, making it a more mature choice.


### Technical issues

There is an issue with the PyPI version of inplace_abn, which accelerates backpropagation for segmentation models. To use it correctly, you need to run a specific script.
```sh
source scripts/03_install_inplace_abn.sh
```

Package Structure
- pipelines/src/pipelines/ – Core implementation of segmentation models.
- pipelines/src/pipelines/diffusion_models/ – Logic for audio separation using diffusion models.
- pipelines/src/pipelines/gan/ – Implementation of CycleGAN for audio source separation.
- scripts/ – Example scripts demonstrating package functionality.
- Makefile – Commands to simplify usage for data scientists.
### Code formatting
- Linter: flake8
- Code Formatter: autopep8

### Type hints and doctsrings
The implementation adheres to the Zen of Python, with a strong emphasis on PEP standards:

- PEP 8 – Style guide for Python code.
- PEP 257 – Docstring conventions.
- PEP 484 – Type hinting.
The code follows best practices in object-oriented programming and design patterns, with modularity and future extensibility in mind.

### Further ideas...
- Extend segmentation models to include Wave-U-Net for improved performance.
- Support a wider range of datasets (currently only .wav file with separate stems).
- Reintroduce model tracking via MLflow or a cheap VPS setup for DVC.
- Explore self-hosted alternatives like MinIO for artifact storage.
- Since moving from DAGsHub to GitHub (due to licensing changes), .dvc files are no longer tracked. Most tracking has been migrated to TensorBoard, which is sufficient for current research. However, if I can set up an MLflow server and an artifact repository (e.g., S3 or MinIO with persistent storage), I may reconsider full tracking and model registry integration.


### References
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [multi-source-diffusion-models](https://github.com/gladia-research-group/multi-source-diffusion-models)
- [audio_diffusion_pytorch](https://github.com/audio-diffusion/audio-diffusion-pytorch)







