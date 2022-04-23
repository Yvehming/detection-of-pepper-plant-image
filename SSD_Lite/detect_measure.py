import os
import pyrealsense2 as rs
import cv2
import numpy as np
from find_root import detect_root
import time
from tflite_runtime.interpreter import Interpreter
import matplotlib.pyplot as plt

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

align_to = rs.stream.color
align = rs.align(align_to)
found_rgb = False

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

if __name__ == "__main__":
    MODEL_NAME = ""
    GRAPH_NAME = "pepper_detect_2cat_0.5mnet.tflite"
    LABELMAP_NAME = "pepper_class.txt"
    min_conf_threshold = 0.5
    # uart = uart.uart()
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
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    # Load image and resize to expected shape [1xHxWx3]
    while True:
        t1 = cv2.getTickCount()

        frames = pipeline.wait_for_frames()
        # RGB图像与深度图像对齐
        aligned_frames = align.process(frames)

        # 得到对齐得图像
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
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

        # 将图片输入模型中
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 提取出结果，矩形框，类别，得分数
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
        object_name = []
        detected_boxes = []
        # print(classes)
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                cv2.rectangle(frame_show, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                cv2.putText(frame_show, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 0), 2,
                            cv2.LINE_AA)
                detected_boxes.append([xmin, ymin, xmax, ymax])
                object_name.append(labels[int(classes[i])])
        print(object_name)
        print(detected_boxes)
        if 'pepper' in object_name:
            if cv2.waitKey(1) == ord('d'):
                try:
                    color_image = update_frame()
                    frame_show = color_image
                    root = detect_root()
                    coordinate = root.find_root(color_image, GRAPH_NAME)
                    # cv2.circle(root.contour_img, (coordinate[0], coordinate[1]), 10, (0, 0, 0), 1)
                    cv2.imshow("roi", root.contour_img)
                    camera_coordinate = np.array(
                        rs.rs2_deproject_pixel_to_point(depth_intrin, [coordinate[0], coordinate[1]],
                                                        rs.depth_frame.get_distance(
                                                            aligned_depth_frame, coordinate[0],
                                                            coordinate[1])))
                    # roi_z = depth_image[root.xmax - 10:root.ymax + 10, root.xmin:root.ymin + 10]
                    # pd.DataFrame(roi_z).to_csv('sample.csv')
                    # f, ax = plt.subplots()
                    # sns.heatmap(roi_z, center=0, square=True, linewidths=1, cmap=plt.get_cmap('Greens'))
                    # plt.savefig("heatmap.svg")
                    # plt.show()
                    # print(roi_z.shape)
                    # print(roi_z[root.bottom[1], root.bottom[0]])
                    temp = camera_coordinate.copy()
                    camera_coordinate[0] = temp[0]
                    camera_coordinate[1] = temp[2]
                    camera_coordinate[2] = -temp[1]
                    
                    print(camera_coordinate * 100)
                    print("测距结束")
                    cv2.circle(frame_show, (coordinate[0], coordinate[1]), 10, (225, 0, 0), 1)
                    cv2.imshow("color_image", frame_show)
                    key = cv2.waitKey(0)
                    if key == 27:
                        cv2.destroyAllWindows()
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
                except IndexError:
                    print("未检测到ROI")
                    raise
        cv2.imshow("color_image", frame_show)
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1
        # key = cv2.waitKey(0)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        # Press any key to continue to next image, or press 'q' to quit

