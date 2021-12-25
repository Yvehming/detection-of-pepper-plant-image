# TensorFlow Lite on Raspberry-Pi
## Introduction
This repository contains tflite model that detects objects, and the process of building tflite model.
### Prerequisite
TensorFlow-GPU version 1.13, TensorFlow Object Detection API version 1.13.

All the operations are based on Windows 10 system.

You can install tensorflow-gpu and its object detection API by referring this page: 
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/

To proceed smoothly, you should pay attention to some slight differences:

1. The official tutorial recommends tensorflow 1.14, but tensorflow 1.13 is a better choice when converting frozen inference 
graph to tflite.
   
2. You can install CuDA and cudnn using conda command, like :

`conda install cudatoollit=10.0`
   
`conda install cudnn`

The above commands will install CuDA and cudnn in your Anaconda virtual environment.
   
## Train models on Windows 10
After install TensorFlow Object Detection API, 

## Convert to tflite
To convert checkpoint to tflite, we need to create a new virtual environment and build tensorflow from source
## Reference
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
https://github.com/tensorflow/models/tree/v1.13
   
