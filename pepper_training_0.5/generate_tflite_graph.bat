echo start
E:
cd E:\Tensorflow_API_1.13\research\object_detection
set CONFIG_FILE=E:\Tensorflow_API_1.13\research\object_detection\pepper_training_0.5\ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync.config
set CHECKPOINT_PATH=E:\Tensorflow_API_1.13\research\object_detection\pepper_training_0.5\training\model.ckpt-xxxxx
set OUTPUT_DIR=E:\Tensorflow_API_1.13\research\object_detection\pepper_training_0.5
call conda activate TF1.13
call python export_tflite_ssd_graph.py --pipeline_config_path=%CONFIG_FILE% --trained_checkpoint_prefix=%CHECKPOINT_PATH% --output_directory=%OUTPUT_DIR% --add_postprocessing_op=true