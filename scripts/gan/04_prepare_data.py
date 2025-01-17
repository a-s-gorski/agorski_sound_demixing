import lmdb
import numpy as np

# from pipelines.gan.create_lmdb_dataset import create_lmdb
from pipelines.gan import datanum_pb2
from pipelines.gan.prepare.lmdb_utils import create_lmdb
from pipelines.gan.config import load_gan_config

parent_folder = 'data/train'

config = load_gan_config('configs/gan/config.yaml')

create_lmdb(parent_folder, 'musdb_train', config=config)

parent_folder = 'data/test'

create_lmdb(parent_folder, 'musdb_test', False, config=config)

for lmdb_name in ['musdb_train', 'musdb_test', 'musdb_valid']:
    env = lmdb.open(lmdb_name, readonly=True)
    with env.begin() as txn:
        raw_datum = txn.get(b'00000000')

    datum = datanum_pb2.DataNum()
    datum.ParseFromString(raw_datum)

    mixture = np.fromstring(datum.mixture, dtype=np.float32)
    vocals = np.fromstring(datum.vocals, dtype=np.float32)
    vocals_indices = np.fromstring(datum.vocals_indices, dtype=np.int32)
    print(mixture.shape)
    print(vocals.shape)
    print(vocals_indices)

