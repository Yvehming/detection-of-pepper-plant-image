from TFLite_detection_image import tflite_image_detection
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# https://blog.csdn.net/lights_joy/article/details/46291229

def find_root(image_path):
    box = []
    image = cv2.imread(image_path)
    class_path = 'pepper_class.txt'
    model_path = 'pepper_detect_2cat_v3.tflite'
    result = tflite_image_detection(class_path, model_path, image_path)
    for i in range(len(result)):
        if str(result[i][0]) == 'root':
            box = result[i]
            box.remove(box[0])
    src = image[box[1] - 10:box[3] + 10, box[0]:box[2] + 10]
    h, w, _ = src.shape
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 1.8 * g - b - r
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(np.array(gray_u8, dtype=np.uint8), -1.0, 255, cv2.THRESH_OTSU)
    hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 定义结构元素
    dilate = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    contour_img = opening.copy()
    _, contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(contour_img, [contours[k]], 0)
    _, contours_1, hierarchy_1 = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours_1[0]
    bottommost = list(cnt[cnt[:, :, 1].argmax()][0])
    bottommost[0] += box[0]
    bottommost[1] += box[1]
    return  bottommost
if __name__ == "__main__":
    files = os.listdir('img')
    for file in files:
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
        print(box)
        # box = [259, 313, 295, 370]
        src = cv2.imread(image_path)
        # # src = cv2.GaussianBlur(src, (3, 3), 0)  # 高斯滤波
        src = src[box[1]-10:box[3]+10, box[0]:box[2]+10]
        h, w, _ = src.shape
        fsrc = np.array(src, dtype=np.float32) / 255.0
        (b, g, r) = cv2.split(fsrc)
        gray = 1.8 * g - b - r
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
        (thresh, bin_img) = cv2.threshold(np.array(gray_u8, dtype=np.uint8), -1.0, 255, cv2.THRESH_OTSU)
        hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
        plt.plot(hist)
        # plt.savefig("curve.svg")
        plt.show()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 定义结构元素
        dilate = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)
        closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        contour_img = opening.copy()
        _, contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area = []
        for j in range(len(contours)):
            area.append(cv2.contourArea(contours[j]))
        max_idx = np.argmax(area)
        max_area = cv2.contourArea(contours[max_idx])
        for k in range(len(contours)):
            if k != max_idx:
                cv2.fillPoly(contour_img, [contours[k]], 0)
        # for i in range(2):
        #     closing = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel)
        origin = cv2.imread(image_path)
        cv2.imshow("origin", src)
        # cv2.imwrite("origin.jpg", src)
        cv2.imshow("bin",bin_img)
        cv2.imshow("morph", opening)
        # cv2.imwrite("morph.jpg", opening)
        _, contours_1, hierarchy_1 = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = contours_1[0]
        bottommost = list(cnt[cnt[:, :, 1].argmax()][0])
        print(bottommost)
        cv2.circle(origin, (bottommost[0]+box[0], bottommost[1]+box[1]), 10, (225, 0, 0), 1)
        cv2.imshow("original position", origin)
        print(find_root(image_path))
        cv2.circle(contour_img, (bottommost[0], bottommost[1]), 10, (225, 0, 0), 1)
        cv2.imshow("result", contour_img)
        # cv2.imwrite("result.jpg",contour_img)
        # cv2.imshow("result", max_area)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
