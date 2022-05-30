# 基于Tensorflow Lite的辣椒苗识别
同时识别辣椒植株及其感兴趣区。

所用模型：SSD_0.5MobileNetv1，由TensorFlow Object Detection API训练和转化得到。

转换为tflite前mAP = 88.03%。

转换为tflite后mAP = 84.12%。

![](img/detected.jpg)