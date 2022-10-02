# SFPN: Synthetic FPN for Object Detection

![](https://img.shields.io/badge/Python-3-blue)
![](https://img.shields.io/badge/TensorFlow-2-orange)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This project proposes a new SFPN (Synthetic Fusion Pyramid Network) arichtecture which creates various synthetic layers between layers of the original FPN to enhance the accuracy of light-weight CNN backones to extract objects' visual features more accurately.

<div align=center>
<img src=https://github.com/D0352276/SFPN-Synthetic-FPN-for-Object-Detection/blob/main/demo/demo.png width=100% />
</div>

The experiments prove the SFPN architecture outperforms either the large backbone VGG16, ResNet50 or light-weight backbones such as MobilenetV2 based on AP score.

Paper Link: https://arxiv.org/abs/2203.02445

This paper has been accepted by IEEE ICIP 2022.


## Requirements

- [Python 3](https://www.python.org/)
- [TensorFlow 2](https://www.tensorflow.org/)
- [OpenCV](https://docs.opencv.org/4.5.2/d6/d00/tutorial_py_root.html)
- [Numpy](http://www.numpy.org/)

## How to Reproduce the Result?
```bash
#Train
python3 main.py -t cfg/train_coco.cfg

#MS-COCO Evaluation
python3 main.py -ce cfg/m2_224_3.cfg

#FPS Evaluation
python3 main.py -fe cfg/m2_224_3.cfg

#Drawing Confidence Figure
python3 main.py -pc cfg/m2_224_3.cfg

```
The cfg file can be changed by the paper prosoed versions. It has options like "m2_224_3.cfg", "m2_320_5.cfg", ......, and "vgg_320_9.cfg".

##Fully Model Weights Files
It is to be released later !

### Fully Dataset
The entire MS-COCO dataset is too large, here only a few pictures are stored for DEMO, 

if you need complete data, please download on this [page.](https://cocodataset.org/#download)

