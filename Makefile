fix_ffmpeg:
	sudo apt-get install -y ffmpeg
	pip uninstall -y ffmpeg-python==0.2.0
	pip install ffmpeg-python==0.2.0
