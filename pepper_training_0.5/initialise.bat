echo start
E:
cd E:\Tensorflow_API_1.13\research\object_detection\pepper_training_0.5
call conda activate TF1.13
call python model_main.py --alsologtostderr --model_dir=training/ --pipeline_config_path=ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync.config