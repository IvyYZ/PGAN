![Python 3](https://img.shields.io/badge/python-3-green.svg) ![Pytorch 0.3](https://img.shields.io/badge/pytorch-0.3-blue.svg)
#RPGAN
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
- Download datasets through the links below, and `unzip`.  
*Market1501*:[[Baidu Pan]link](https://pan.baidu.com/s/1XN5EyIFHcOxATezcWbVLTA)  

### Model
- Create directories for datasets:
```
mkdir bestmodel
cd bestmodel
```
- Download trained model through the links below. it include encoder pre-model and the whole model.  
*encoder pre-trained model*:[[Baidu Pan]link](https://pan.baidu.com/s/1T626WlYHoad31Kadn2x1MA)  
*GAN model*:[[Baidu Pan]link](https://pan.baidu.com/s/1haO4CvBFGuK6QA9BuStwzA) 

## Run code
defalut: +reranking, if you want to remove re_ranking, you need to modify ./reid/evaluators.py
```
sh train.sh
```
## Citation
-If you use this method or this code in your research, please cite as:
@article{zhang2020pgan,
  title={PGAN: Part-based nondirect coupling embedded GAN for person reidentification},
  author={Zhang, Yue and Jin, Yi and Chen, Jianqiang and Kan, Shichao and Cen, Yigang and Cao, Qi},
  journal={IEEE MultiMedia},
  volume={27},
  number={3},
  pages={23--33},
  year={2020},
  publisher={IEEE}
}

## Acknowledgements
Our code is inspired by [FDGAN] (https://github.com/yxgeee/FD-GAN), [PCB] (https://github.com/syfafterzy/PCB_RPP_for_reIDand) and [open-reid](https://github.com/Cysu/open-reid).


