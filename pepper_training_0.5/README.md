本文件夹是tensorflow object detection api训练SSD_0.5MobileNetv1，并转换为tflite格式的相关配置文件。

需要安装tensorflow object detection api进行模型的训练和转换。

.config文件规定了模型的训练、评估相关的相关参数，以及模型本身的相关参数设置。包括预训练模型、数据增强方法、优化器、批尺寸、训练步数等。

运行initialise.bat开始训练，model_main.py中第62行config = tf.estimator.RunConfig()规定了检查点的保存周期和评估周期。

依次运行generate_tflite_graph.bat和convert.bat生成.tflite文件，注意修改.bat文件中的相关路径。