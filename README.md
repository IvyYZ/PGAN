![Python 3](https://img.shields.io/badge/python-3-green.svg) ![Pytorch 0.3](https://img.shields.io/badge/pytorch-0.3-blue.svg)
# DIGAN
ReID 
## Prerequisites
- Python 3
- [Pytorch](https://pytorch.org/) (We run the code under version 0.3.1, maybe lower versions also work.)

## Getting Started

### Installation
- Install dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). You can install all the dependencies by:
```
pip install scipy, pillow, torchvision, sklearn, h5py, dominate, visdom
```
### Datasets
- Create directories for datasets:
```
mkdir datasets
cd datasets/
```
- Download datasets through the links below, and `unzip` them in the same root path.  
*Market1501*:[[Baidu Pan]](https://pan.baidu.com/s/1XN5EyIFHcOxATezcWbVLTA)  

### Model
- Create directories for datasets:
```
mkdir bestmodel
cd bestmodel
```
- Download trained model through the links below. it include encoder pre-model and the whole model.  
*encoder pre-trained model*:[[Baidu Pan]](https://pan.baidu.com/s/1T626WlYHoad31Kadn2x1MA)  
*GAN model*:[[Baidu Pan]](https://pan.baidu.com/s/1haO4CvBFGuK6QA9BuStwzA) 

## Run code
```
sh train.sh
```

## Acknowledgements
Our code is inspired by [FDGAN] (https://github.com/yxgeee/FD-GAN), [PCB] (https://github.com/syfafterzy/PCB_RPP_for_reIDand) and [open-reid](https://github.com/Cysu/open-reid).


