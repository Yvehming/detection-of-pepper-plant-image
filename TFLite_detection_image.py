import os
from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np

# 检测图片函数
def tflite_image_detection(class_path, model_path, image_path):
    class_box = []
    output = []
    min_conf_threshold = 0.5
    image_path = image_path

    # Load the label map
    with open(class_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model.
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Loop over every image and perform detection

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image = cv2.GaussianBlur(image, (5, 5), 1)  # 高斯滤波
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    print(scores)
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            class_box.append(object_name)
            class_box.append(xmin)
            class_box.append(ymin)
            class_box.append(xmax)
            class_box.append(ymax)
            output.append(class_box)
            class_box = []
    return output

# 检测深度相机的读取到的图片
def tflite_video_detection(class_path, model_path, image):
    class_box = []
    output = []
    min_conf_threshold = 0.5

    # Path to .tflite file, which contains the model that is used for object detection


    # Load the label map
    with open(class_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model.
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Loop over every image and perform detection

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.GaussianBlur(image, (5, 5), 1)  # 高斯滤波
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)


    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    # print(scores)
    # print(classes)
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            class_box.append(object_name)
            class_box.append(xmin)
            class_box.append(ymin)
            class_box.append(xmax)
            class_box.append(ymax)
            output.append(class_box)
            class_box = []
    return output

# 检测图片
if __name__ == "__main__":
    MODEL_NAME = ""
    GRAPH_NAME = "pepper_detect_2cat_0.5mnet.tflite"
    LABELMAP_NAME = "pepper_class.txt"
    min_conf_threshold = 0.5
    use_TPU = False
    image_path = 'img/test7.jpg'

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library

    from tflite_runtime.interpreter import Interpreter

    # Load the label map
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

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Loop over every image and perform detection

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image = cv2.GaussianBlur(image, (5, 5), 1)  # 高斯滤波
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    print(scores)
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            # cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
            #               (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
            #               cv2.FILLED)  # Draw white box to put label text in
            # cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
            #             2)  # Draw label text

    # All the results have been drawn on the image, now display the image
    cv2.imshow('Object detector', image)
    cv2.imwrite('result.jpg', image)
    # Press any key to continue to next image, or press 'q' to quit
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()