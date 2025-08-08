
# MobileNet SSDv2 Object Detection

Mobilenet-SSDv2: An improved object detection model for embedded systems.   
This repository uses [PyTorch](https://pytorch.org/) and [OpenMMLab](https://openmmlab.com/) frameworks.   
[Paper](https://ieeexplore.ieee.org/abstract/document/9219319)

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

### 3. Install MMDetection 
```bash
pip install -e .
```
---

## Pascal VOC

Network|size |mAP|config|weight
:---:|:---:|:---:|:---:|:---:|
MobileNet-SSD|512|?|[link](https://github.com/vimyc95/Mobilenet-SSDv2/blob/main/projects/MobilenetSSDv2/configs/ssdlite512_mobilenetv2_4xb8-200e_voc.py)|[download](https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view?usp=sharing&resourcekey=0-1Lpfs4EvGDeCQz12AF64hQ)
MobileNet-SSDv2|512|74.9|[link](https://github.com/vimyc95/Mobilenet-SSDv2/blob/main/projects/MobilenetSSDv2/configs/ssdlite512_fpn_mobilenetv2_4xb8-200e_voc.py)|[download](https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view?usp=sharing&resourcekey=0-1Lpfs4EvGDeCQz12AF64hQ)

---


## Usage

### Train
```bash
./tools/dist_train.sh projects/MobilenetSSDv2/configs/ssdlite512_fpn_mobilenetv2_4xb8-200e_voc.py ${GPUS}
```

### Inference
```bash
./tools/dist_test.sh ${config} ${ckpt} ${GPUS}
```

---



