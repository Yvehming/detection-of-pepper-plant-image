from TFLite_detection_image import tflite_image_detection
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
files = os.listdir('img')
for file in files:
    print(file)
    box = []
    image_path = 'img/' + file
    class_path = 'pepper_class.txt'
    model_path = 'pepper_detect_2cat_v3.tflite'
    result = tflite_image_detection(class_path, model_path, image_path)
    print(result)
    for i in range(len(result)):
        if str(result[i][0]) == 'root':
            box = result[i]
            box.remove(box[0])
    # box = [259, 313, 295, 370]
    src = cv2.imread(image_path)
    src = src[box[1]:box[3] , box[0]:box[2] ]
    src = cv2.GaussianBlur(src, (3, 3), 1)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    img = src
    Z = src.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    imgHSV = img
    # lower = np.array([14, 150, 40])
    lower = np.array(center.min(0), dtype='int32')
    upper = np.array([255, 255, 255])
    # 获得指定颜色范围内的掩码
    mask = cv2.inRange(imgHSV, lower, upper)
    # 对原图图像进行按位与的操作，掩码区域保留
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    contour_img = mask.copy()
    _, contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(contour_img, [contours[k]], 0)
    cv2.imshow("origin", cv2.cvtColor(src, cv2.COLOR_HSV2BGR))
    cv2.imshow("Mask", mask)
    #显示分割后的图像
    cv2.imshow("Result", imgResult)
    cv2.imshow("contour", contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

