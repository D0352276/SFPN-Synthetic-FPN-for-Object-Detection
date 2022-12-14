# SFPN: Synthetic FPN for Object Detection

![](https://img.shields.io/badge/Python-3-blue)
![](https://img.shields.io/badge/TensorFlow-2-orange)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This project was developed from CSL-YOLO and is its follow-up study. The proposed SFPN (Synthetic Fusion Pyramid Network) arichtecture creates various synthetic layers between layers of the original FPN to enhance the accuracy of light-weight CNN backones to extract objects' visual features more accurately.

This paper has been accepted by IEEE ICIP 2022.

Paper Link: https://arxiv.org/abs/2203.02445

<div align=center>
<img src=https://github.com/D0352276/SFPN-Synthetic-FPN-for-Object-Detection/blob/main/demo/sfmandresulttable.png width=100% />
</div>

Architecture of Synthetic Fusion Module (SFM). The blue, orange and gray arrows represent the optional inputs. The experiments prove the SFPN architecture outperforms either the large backbone VGG16, ResNet50 or light-weight backbones such as MobilenetV2 based on AP score.

<div align=center>
<img src=https://github.com/D0352276/SFPN-Synthetic-FPN-for-Object-Detection/blob/main/demo/sfpn.png width=100% />
</div>

Synthetic Fusion Pyramid Network with synthetic outputs layers: (a) SFPN-5-SOL, (b) SFPN-9-SOL. The yellow squares represent SFBs.


## More Suitable Feature

<div align=center>
<img src=https://github.com/D0352276/SFPN-Synthetic-FPN-for-Object-Detection/blob/main/demo/moresuitfeats.png width=70% />
</div>

In this example, we found that the synthetic layers fit some objects better than original layers. The top three frames are the confidence maps output by SFPN-3, and the bottom three frames are the confidence maps output by SFPN-5-SOL.


## Requirements

- [Python 3](https://www.python.org/)
- [TensorFlow 2](https://www.tensorflow.org/)
- [OpenCV](https://docs.opencv.org/4.5.2/d6/d00/tutorial_py_root.html)
- [Numpy](http://www.numpy.org/)

## How to Reproduce the Results ?
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


## Fully Model Weights Files
[Download Link](https://drive.google.com/drive/folders/1aF-nK44huxbiQP1zZsg8Mjwfr0n2Nz-k?usp=sharing)

This link contains all weights of CNN models mentioned in the paper.


## Fully Dataset
The entire MS-COCO dataset is too large, here only a few pictures are stored for DEMO, 

if you need complete data, please download on this [page.](https://cocodataset.org/#download)

## Results
 Comparison of SFPN-5 and SFPN-9 with the baseline (SFPN-3) on MS-COCO. Note that the FPS is derived from the I7-6600U.
 
<div align=center>
<img src=https://github.com/D0352276/SFPN-Synthetic-FPN-for-Object-Detection/blob/main/demo/results.png width=60% />
</div>


