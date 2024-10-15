fix_ffmpeg:
	sudo apt-get install -y ffmpeg
	pip uninstall -y ffmpeg-python==0.2.0
	pip install ffmpeg-python==0.2.0


format_package:
	pip install isort autopep8
	isort pipelines/src/pipelines
	autopep8 --recursive --in-place pipelines/src/pipelines

train_accompaniment_vocals_mobilenet_subbandtime:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,mobilenet_subbandtime.yaml"

train_accompaniment_vocals_resunet_ismir2021:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,resunet_ismir2021.yaml"

train_accompaniment_vocals_resunet_subbandtime:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,resunet_subbandtime.yaml"

train_accompaniment_vocals_resunet:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,resunet.yaml"

train_accompaniment_vocals_unet:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,unet.yaml"

train_vocals_accompaniment_mobilenet_subbandtime:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,mobilenet_subbandtime.yaml"

train_vocals_accompaniment_resunet_ismir2021:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,resunet_ismir2021.yaml"

train_vocals_accompaniment_resunet_subbandtime:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,resunet_subbandtime.yaml"

train_vocals_accompaniment_resunet:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,resunet.yaml"

train_vocals_accompaniment_unet_subbandtime:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,unet_subbandtime.yaml"

train_vocals_accompaniment_unet:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,unet.yaml"

train_vocals_bass_drums_other_resunet_subbandtime:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-bass-drums-other,resunet_subbandtime.yaml"

train_vocals_bass_drums_other_unet:
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-bass-drums-other,unet.yaml"
