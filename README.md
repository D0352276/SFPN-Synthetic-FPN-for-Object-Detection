# SFPN: Synthetic FPN for Object Detection

![](https://img.shields.io/badge/Python-3-blue)
![](https://img.shields.io/badge/TensorFlow-2-orange)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This project proposes a new SFPN (Synthetic Fusion Pyramid Network) arichtecture which creates various synthetic layers between layers of the original FPN to enhance the accuracy of light-weight CNN backones to extract objects' visual features more accurately.

<div align=center>
<img src=https://github.com/D0352276/SFPN-Synthetic-FPN-for-Object-Detection/blob/main/demo/results.png width=100% />
</div>

The experiments prove the SFPN architecture outperforms either the large backbone VGG16, ResNet50 or light-weight backbones such as MobilenetV2 based on AP score.

Paper Link: https://arxiv.org/abs/2203.02445

This paper has been accepted by IEEE ICIP 2022.


## Requirements

- [Python 3](https://www.python.org/)
- [TensorFlow 2](https://www.tensorflow.org/)
- [OpenCV](https://docs.opencv.org/4.5.2/d6/d00/tutorial_py_root.html)
- [Numpy](http://www.numpy.org/)


