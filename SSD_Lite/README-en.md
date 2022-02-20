# TensorFlow Lite on Raspberry-Pi
## Introduction
This repository contains tflite model that detects objects, and the process of building tflite model.
### Prerequisite
TensorFlow-GPU version 1.13, [TensorFlow Object Detection API version 1.13](https://github.com/tensorflow/models/tree/v1.13.0).

All operations are based on Windows 10 system.

You can install tensorflow-gpu and its object detection API by referring this page: 
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/

To proceed smoothly, you should pay attention to these below:

1. The official tutorial recommends tensorflow 1.14, but tensorflow 1.13 is a better choice when converting frozen inference 
graph to tflite.
   
2. You can install CuDA and cudnn using conda command, like :

   `conda install cudatoollit=10.0`
   
   `conda install cudnn`

   The above commands will install CuDA and cudnn in your Anaconda virtual environment.

3. To install TensorFlow Object Detection API successfully, protobuf should be installed on your comuter.

4. According to the [TensorFlow model document](https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md)
   , using toco to transfer of TensorFlow Lite model should install bazel and [Visual C++ build tools 2015](https://download.microsoft.com/download/5/F/7/5F7ACAEB-8363-451F-9425-68A90F98B238/visualcppbuildtools_full.exe) and MSYS2 and build [TensorFlow from source](https://www.tensorflow.org/install/install_sources)
   .When building TensorFlow from source, the compiling process can be use a lot of RAM. If your system is memory-constrained, limit Bazel's RAM usage with: `--local_ram_resources=2048`.
   
## Train models on Windows 10
After install TensorFlow Object Detection API, 

## Convert to tflite
To convert checkpoint to tflite, we need to create a new virtual environment and build tensorflow from source
## Reference
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

https://github.com/tensorflow/models/tree/v1.13
   
