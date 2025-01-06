#!/bin/bash

# Check if conda command is available
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Installing Miniconda..."

    # Download the Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    
    # Install Miniconda
    bash Miniconda3-latest-Linux-x86_64.sh -b  # -b flag for silent install (without prompts)

    # Initialize conda and update .bashrc
    ~/miniconda3/bin/conda init bash

    # Source .bashrc to make conda available immediately
    source ~/.bashrc

    rm Miniconda3-latest-Linux-x86_64.sh

    echo "Miniconda installed and initialized."
else
    echo "Conda is already installed."
fi

# Create a new conda environment with Python 3.8 if it doesn't exist
ENV_NAME="master"
if conda env list | grep -q "$ENV_NAME"
then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating a new environment with Python 3.8..."
    conda create -y -n $ENV_NAME python=3.8
    echo "Environment '$ENV_NAME' created."
fi

# Activate the environment
echo "Activating the environment '$ENV_NAME'..."
conda activate $ENV_NAME
