from TFLite_detection_image import tflite_video_detection
from TFLite_detection_image import tflite_image_detection
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# https://blog.csdn.net/lights_joy/article/details/46291229

class detect_root():
    def __init__(self):
        self.class_path = 'pepper_class.txt'
    def find_root(self, image, model_path):
        box = []
        image = image
        class_path = self.class_path
        result = tflite_video_detection(class_path, model_path, image)
        class_name = []
        for i in range(len(result)):
            class_name.append(result[i][0])
        while 'root' not in class_name:
            result = tflite_video_detection(class_path, model_path, image)
            class_name = []
            for i in range(len(result)):
                class_name.append(result[i][0])
        print(result)
        for i in range(len(result)):
            if str(result[i][0]) == 'root':
                if result[i][1] + result[i][3] >300:
                    box = result[i]
                    box.remove(box[0])
        self.xmin = box.copy()[0]
        self.xmax = box.copy()[1]
        self.ymin = box.copy()[2]
        self.ymax = box.copy()[3]
        src = image[box[1] - 10:box[3] + 10, box[0]:box[2] + 10]
        h, w, _ = src.shape
        fsrc = np.array(src, dtype=np.float32) / 255.0
        (b, g, r) = cv2.split(fsrc)
        gray = 1.8 * g - b - r
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
        (thresh, bin_img) = cv2.threshold(np.array(gray_u8, dtype=np.uint8), -1.0, 255, cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 定义结构元素
        dilate = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)
        closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        self.contour_img = opening.copy()
        _, contours, hierarchy = cv2.findContours(self.contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        area = []
        for j in range(len(contours)):
            area.append(cv2.contourArea(contours[j]))
        max_idx = np.argmax(area)
        max_area = cv2.contourArea(contours[max_idx])
        for k in range(len(contours)):
            if k != max_idx:
                cv2.fillPoly(self.contour_img, [contours[k]], 0)
        _, contours_1, hierarchy_1 = cv2.findContours(self.contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = contours_1[0]
        self.bottom = list(cnt[cnt[:, :, 1].argmax()][0])
        # print(self.bottom)
        bottommost = self.bottom.copy()
        bottommost[0] += box[0]
        bottommost[1] = box[1] + bottommost[1] - 10
        return bottommost
    def post_process(self, z):
        pass



if __name__ == "__main__":
    files = os.listdir('img')
    for file in files:
        box = []
        image_path = 'img/' + file
        class_path = 'pepper_class.txt'
        model_path = 'pepper_detect_2cat_0.5mnet.tflite'
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
        cv2.imwrite("gray.jpg", gray_u8)
        cv2.imwrite("origin.jpg", src)
        cv2.imshow("bin", bin_img)
        cv2.imshow("morph", opening)
        # cv2.imwrite("morph.jpg", opening)
        _, contours_1, hierarchy_1 = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = contours_1[0]
        bottommost = list(cnt[cnt[:, :, 1].argmax()][0])
        print(bottommost)
        cv2.circle(origin, (bottommost[0]+box[0], bottommost[1]+box[1]-10), 10, (225, 0, 0), 1)
        cv2.imshow("original position", origin)
        # print(find_root(image_path))
        cv2.circle(contour_img, (bottommost[0], bottommost[1]), 10, (225, 0, 0), 1)
        cv2.imshow("result", contour_img)
        # cv2.imwrite("result.jpg",contour_img)
        cv2.imshow("result", max_area)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
