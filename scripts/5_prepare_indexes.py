import argparse
import os
import pickle
from typing import NoReturn
import yaml

from typing import Dict

import h5py

from pipelines.utils.process_audio import create_indexes

workspace = "."

INDEXES_CONFIG_YAML_1 = "./configs/indexing/sr=44100,vocals-accompaniment.yaml"
INDEXES_CONFIG_YAML_2 = "./configs/indexing/sr=44100,vocals-bass-drums-other.yaml"

create_indexes(workspace, INDEXES_CONFIG_YAML_1)
create_indexes(workspace, INDEXES_CONFIG_YAML_2)