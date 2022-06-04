import os
import pyrealsense2 as rs
import cv2
import numpy as np
from find_root import detect_root
import realsense_init
import uart
import time
from tflite_runtime.interpreter import Interpreter
import matplotlib.pyplot as plt
# 与小车通信实时识别和定位
### realsense的图像噪声较大，对根部感兴趣区的获取存在一定问题，要用realsense拍摄图片，增加数据集

if __name__ == "__main__":
    MODEL_NAME = ""
    GRAPH_NAME = "pepper_detect_2cat_0.5mnet.tflite"
    LABELMAP_NAME = "pepper_class.txt"
    min_conf_threshold = 0.5
    camera = realsense_init.camera()
    uart = uart.uart()
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
   
    # Load image and resize to expected shape [1xHxWx3]
    while True:
        color_image = camera.read_rgb_image()
        frame_show = color_image
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        # depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
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
                detected_boxes.append([xmin, ymin, xmax, ymax])
                object_name.append(labels[int(classes[i])])
#         print(object_name)
#         print(detected_boxes)
        if 'pepper' in object_name and 'root' in object_name:
            if 400 < (detected_boxes[object_name.index('pepper')][0] + detected_boxes[object_name.index('pepper')][2])/2 < 500:
#             if True:
#                 uart.send_data('[detected]')
#                 if cv2.waitKey(1) == ord('d'):
                while uart.read_data(3).decode('ascii') != 'ACK':
                    uart.send_data('[1]')
                print('ACK received')
                while uart.read_data(3).decode('ascii') != '666':
                    pass
                print('666 received')
                time.sleep(1.0)
                try:
                    color_image = camera.read_aligned_image()
                    frame_show = color_image
                    root = detect_root()
                    coordinate = root.find_root(color_image, GRAPH_NAME)
                    # cv2.circle(root.contour_img, (coordinate[0], coordinate[1]), 10, (0, 0, 0), 1)
                    camera_coordinate = np.array(
                        rs.rs2_deproject_pixel_to_point(camera.depth_intrin, [coordinate[0], coordinate[1]],
                                                        rs.depth_frame.get_distance(
                                                            camera.aligned_depth_frame, coordinate[0],
                                                            coordinate[1])))
                    while camera_coordinate[0] == 0 and camera_coordinate[1] == 0:
                        color_image = camera.read_aligned_image()
                        frame_show = color_image
                        coordinate = root.find_root(color_image, GRAPH_NAME)
                        # cv2.circle(root.contour_img, (coordinate[0], coordinate[1]), 10, (0, 0, 0), 1)
                        # cv2.imshow("roi", root.contour_img)
                        camera_coordinate = np.array(
                            rs.rs2_deproject_pixel_to_point(camera.depth_intrin, [coordinate[0], coordinate[1]],
                                                            rs.depth_frame.get_distance(
                                                                camera.aligned_depth_frame, coordinate[0],
                                                                coordinate[1])))
                    
                    temp = camera_coordinate.copy()
                    camera_coordinate[0] = temp[0] -0.03
                    camera_coordinate[1] = temp[2]
                    camera_coordinate[2] = -temp[1]
                    camera_coordinate *= 1000
#                     camera_coordinate = [10.00,200.00,-40.00]
#                     camera_coordinate[0] = 10.000
#                     camera_coordinate[1] = 210.000
#                     camera_coordinate[2] = -20.000
                    print('坐标值:', camera_coordinate)
                    print("测距结束")
                    uart.send_data('[' + '%.2f' % camera_coordinate[0] + ']')
                    time.sleep(0.1)
                    uart.send_data('[' + '%.2f' % camera_coordinate[1] + ']')
                    time.sleep(0.1)
                    uart.send_data('[' + '%.2f' % camera_coordinate[2] + ']')
#                     cv2.circle(frame_show, (coordinate[0], coordinate[1]), 10, (225, 0, 0), 1)
#                     cv2.imshow("roi", root.contour_img)
#                     cv2.imshow("color_image", frame_show)
                    while uart.read_data(2).decode('ascii') != 'GO':
                        pass
                    time.sleep(1.0)
                    print('重新检测')
#                     key = cv2.waitKey(0)
#                     if key == 27:
#                         cv2.destroyAllWindows()
#                     if key == ord('q'):
#                         cv2.destroyAllWindows()
#                         break
                except IndexError:
                    print("未检测到ROI")
                    raise
        cv2.imshow("color_image", frame_show)
        # key = cv2.waitKey(0)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        # Press any key to continue to next image, or press 'q' to quit


