import os
import pyrealsense2 as rs
import cv2
import numpy as np
from find_root import find_root
import time
from tflite_runtime.interpreter import Interpreter

def update_frame():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    return color_image

### realsense的图像噪声较大，对根部感兴趣区的获取存在一定问题，要用realsense拍摄图片，增加数据集
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

align_to = rs.stream.color
align = rs.align(align_to)
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

if __name__ == "__main__":
    MODEL_NAME = ""
    GRAPH_NAME = "pepper_detect_2cat_0.5mnet.tflite"
    LABELMAP_NAME = "pepper_class.txt"
    min_conf_threshold = 0.5

    # 导入模型和类别
    with open(LABELMAP_NAME, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model.
    interpreter = Interpreter(model_path=GRAPH_NAME)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Loop over every image and perform detection
    time.sleep(0.5)
    # Load image and resize to expected shape [1xHxWx3]
    while True:
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        frame_show = color_image
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = color_image.shape
        image_resized = cv2.resize(color_image, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)
        # frame_show = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
        # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        # print(classes)

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                if classes[i] == 0:
                    try:
                        color_image = update_frame()
                        frame_show = color_image
                        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                        coordinate = find_root(color_image, GRAPH_NAME)
                        camera_coordinate = np.array(
                            rs.rs2_deproject_pixel_to_point(depth_intrin, [coordinate[0], coordinate[1]],
                                                            rs.depth_frame.get_distance(
                                                                aligned_depth_frame, coordinate[0],
                                                                coordinate[1])))
                        print(camera_coordinate * 100)
                        print("测距结束")
                        cv2.circle(frame_show, (coordinate[0], coordinate[1]), 10, (225, 0, 0), 1)
                        cv2.imshow("color_image", frame_show)
                        key = cv2.waitKey(0)
                        if key == 27:
                            cv2.destroyAllWindows()
                            break
                    except IndexError:
                        print("未检测到ROI")
                        raise

        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        # Press any key to continue to next image, or press 'q' to quit

        # Clean up
    # cv2.destroyAllWindows()
