
# MobileNet SSDv2 Object Detection

Mobilenet-SSDv2[[pdf]](https://ieeexplore.ieee.org/abstract/document/9219319): An improved object detection model for embedded systems.   
This repository uses [PyTorch](https://pytorch.org/) and [OpenMMLab](https://openmmlab.com/) frameworks.

## Installation


### 1. Create and activate a Conda environment
```
# Create a Conda environment with Python 3.10
conda create -n mobilenetssdv2 python==3.10
conda activate mobilenetssdv2
```

### 2. Install Python packages
```bash
# Install PyTorch with CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# Install OpenMMLab dependencies
pip install mmcv==2.2.0
pip install mmpretrain==1.2.0
```

### 3. Install MMDetection 3.1
```bash
pip install -e .
```
---

## Pascal VOC

Network|size |mAP|Config|Download
:---:|:---:|:---:|:---:|:---:|
MobileNet-SSD|512|72.7|[config](https://github.com/vimyc95/Mobilenet-SSDv2/blob/main/projects/MobilenetSSDv2/configs/ssdlite512_mobilenetv2_4xb8-200e_voc.py)|[model](https://drive.google.com/file/d/16oaXvI-jCqTA4ha8JolqTbSINlAPC_iq/view?usp=sharing) \| [logs](https://drive.google.com/file/d/1o4UEWn45BgN5MuXdS2HcQDeB7t77b9QK/view?usp=sharing)
MobileNet-SSDv2|512|75.6|[config](https://github.com/vimyc95/Mobilenet-SSDv2/blob/main/projects/MobilenetSSDv2/configs/ssdlite512_fpn_mobilenetv2_4xb8-200e_voc.py)|[model](https://drive.google.com/file/d/1E_kmog8ziuqvjOcHgBp6xnAvbJ-Mr0Fp/view?usp=sharing) \| [logs](https://drive.google.com/file/d/1LM8FoSBNJNPlPqki_etx1U-3znFtQp7d/view?usp=sharing)

---


## Usage

### Train
```bash
./tools/dist_train.sh projects/MobilenetSSDv2/configs/ssdlite512_fpn_mobilenetv2_4xb8-200e_voc.py ${GPUS}
```

### Evaluate
```bash
./tools/dist_test.sh ${config} ${ckpt} ${GPUS}
```

---


