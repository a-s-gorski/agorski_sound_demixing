fix_ffmpeg:
	sudo apt-get install -y ffmpeg
	pip uninstall -y ffmpeg-python==0.2.0
	pip install ffmpeg-python==0.2.0

prepare_env:
	source scripts/01_setup_environnment.sh
	source scripts/02_install_gpu.sh
	source scripts

download_data_segmentation:
	source scripts/00_download_data.sh

download_data_gan:
	TODO



format_package:
	pip install isort autopep8
	isort pipelines/src/pipelines
	autopep8 --recursive --in-place pipelines/src/pipelines

train_accompaniment_vocals_mobilenet_subbandtime: # done
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,mobilenet_subbandtime.yaml"

train_accompaniment_vocals_resunet_ismir2021: # done
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,resunet_ismir2021.yaml"

train_accompaniment_vocals_resunet_subbandtime: # done
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,resunet_subbandtime.yaml"

train_accompaniment_vocals_resunet: # done
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,resunet.yaml"

train_accompaniment_vocals_unet: # works to be tested on coltrane
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/accompaniment-vocals,unet.yaml"

train_vocals_accompaniment_mobilenet_subbandtime: # done
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,mobilenet_subbandtime.yaml"

train_vocals_accompaniment_resunet_ismir2021: # done
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,resunet_ismir2021.yaml"

train_vocals_accompaniment_resunet_subbandtime: # done
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,resunet_subbandtime.yaml"

train_vocals_accompaniment_resunet: # in-progress
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,resunet.yaml"

train_vocals_accompaniment_unet: # works to be tested on coltrane
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-accompaniment,unet.yaml"

train_vocals_bass_drums_other_resunet_subbandtime: # works to be tested on coltrane
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-bass-drums-other,resunet_subbandtime.yaml"

train_vocals_bass_drums_other_unet: # works to be tested on coltrane
	python3.8 scripts/08_train.py --gpus=1 --config_yaml="configs/training/vocals-bass-drums-other,unet.yaml"

seperate_file:
	TRAIN_CONFIG=accompaniment-vocals,mobilenet_subbandtime; \
	echo "seperating train for $${TRAIN_CONFIG}"; \
	python3.8 scripts/09_separate.py separate_file \
	    --config_yaml="configs/training/$${TRAIN_CONFIG}.yaml" \
    	--checkpoint_path="checkpoints/musdb18/agorski_sound_demixing/config=$${TRAIN_CONFIG},gpus=1/step=0.pth" \
    	--audio_path="resources/vocals_accompaniment_10s.mp3" \
    	--output_path="sep_results/vocals_accompaniment_10s_sep_vocals.mp3"


seperate_dir_accompaniment_vocals_mobilenet_subbandtime:
	TRAIN_CONFIG=accompaniment-vocals,mobilenet_subbandtime; \
	echo "seperating train for $${TRAIN_CONFIG}"; \
	python3.8 scripts/09_separate.py separate_dir \
	    --config_yaml="configs/training/$${TRAIN_CONFIG}.yaml" \
    	--checkpoint_path="checkpoints/musdb18/agorski_sound_demixing/config=$${TRAIN_CONFIG},gpus=1/step=100000.pth" \
    	--audios_dir="datasets/musdb18/train" \
    	--outputs_dir="sep_results/$${TRAIN_CONFIG}/train"; \
	python3.8 scripts/09_separate.py separate_dir \
	    --config_yaml="configs/training/$${TRAIN_CONFIG}.yaml" \
    	--checkpoint_path="checkpoints/musdb18/agorski_sound_demixing/config=$${TRAIN_CONFIG},gpus=1/step=100000.pth" \
    	--audios_dir="datasets/musdb18/test" \
    	--outputs_dir="sep_results/$${TRAIN_CONFIG}/test"; \

seperate_dir_vocals_accompaniment_mobilenet_subbandtime:
	TRAIN_CONFIG=vocals-accompaniment,mobilenet_subbandtime; \
	echo "seperating train for $${TRAIN_CONFIG}"; \
	python3.8 scripts/09_separate.py separate_dir \
	    --config_yaml="configs/training/$${TRAIN_CONFIG}.yaml" \
    	--checkpoint_path="checkpoints/musdb18/agorski_sound_demixing/config=$${TRAIN_CONFIG},gpus=1/step=300000.pth" \
    	--audios_dir="datasets/musdb18/train" \
    	--outputs_dir="sep_results/$${TRAIN_CONFIG}/train"; \
	python3.8 scripts/09_separate.py separate_dir \
	    --config_yaml="configs/training/$${TRAIN_CONFIG}.yaml" \
    	--checkpoint_path="checkpoints/musdb18/agorski_sound_demixing/config=$${TRAIN_CONFIG},gpus=1/step=300000.pth" \
    	--audios_dir="datasets/musdb18/test" \
    	--outputs_dir="sep_results/$${TRAIN_CONFIG}/test"; \


diffusion_train_model:
	python scripts/diffusion_models/02_train.py --dataset_config=configs/diffusion_model/dataset.yaml \
		--training_config=configs/diffusion_model/training.yaml \
		--training_data_path=data/musdb18_hq/train \
		--validation_data_path=data/musdb18_hq/validation