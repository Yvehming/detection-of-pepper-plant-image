## 数据分析，绘制出ROI区域的深度信息的三维图
import os
import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt
from TFLite_detection_image import tflite_image_detection
import time
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
    use_TPU = False


    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library

    from tflite_runtime.interpreter import Interpreter
    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model.
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Loop over every image and perform detection
    time.sleep(2.0)
    # Load image and resize to expected shape [1xHxWx3]
    while True:
        frames = pipeline.wait_for_frames()
        root_box = []
        x = []
        y = []
        z = []
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = color_image.shape
        image_resized = cv2.resize(color_image, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)
        # frame_show = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        frame_show = color_image
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
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                # cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        cv2.imwrite("detect.jpg", frame_show)

        # coordinate = find_root("detect.jpg")
        box = tflite_image_detection(LABELMAP_NAME, GRAPH_NAME, 'detect.jpg')
        print(box)
        for i in range(len(box)):
            if str(box[i][0]) == 'root':
                root_box = box[i]
                root_box.remove(root_box[0])
        for i in range(root_box[0], root_box[2]+1):
            for j in range(root_box[1], root_box[3]+1):
                roi_coordinate = np.array(
                    rs.rs2_deproject_pixel_to_point(depth_intrin, [i, j],
                                                    rs.depth_frame.get_distance(
                                                        aligned_depth_frame, i, j)))

                # print(roi_coordinate)
                x.append(roi_coordinate[0]*100)
                y.append(roi_coordinate[1]*100)
                z.append(roi_coordinate[2]*100)
        fig = plt.figure()
        print(x)
        print(y)
        print(z)
        # 创建绘图区域
        ax = plt.axes(projection='3d')
        # 构建xyz

        ax.scatter3D(x, y, z)
        ax.set_title('3d Scatter plot')
        plt.savefig("draw.svg")
        plt.show()

        cv2.rectangle(color_image, (root_box[0], root_box[1]), (root_box[2], root_box[3]), (10, 255, 0), 2)
        cv2.imshow("color_image", color_image)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        # Press any key to continue to next image, or press 'q' to quit

        # Clean up
    # cv2.destroyAllWindows()