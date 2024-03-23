# Installation

## Requirements
- Python == 3.10.0
- Numpy
- pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- pip install simplejson
- pip install psutil
- pip install tensorboard
- pip install opencv-Python
- pip install matplotlib
- pip install av
- pip install decord
- pip install pytorchvideo
- pip install einops
- pip install pandas
- if you get Error "ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor", change "functional_tensor" to "functional" in pytorchvideo/transforms/augmentations.py 
