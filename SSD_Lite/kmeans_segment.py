from TFLite_detection_image import tflite_image_detection
import numpy as np
import cv2

box = []
image_path = 'img/test5.jpg'
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
src = src[box[1] - 10:box[3] + 10, box[0]:box[2] + 10]
# img = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
img = src
Z = img.reshape((-1, 3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(img.shape)
print(center)
# blur = cv2.GaussianBlur(res2, (5, 5), 0)
# res3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('res2', res2)
# img_gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', img_gray)
# # blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
# res3, th3 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
# cv2.imshow('res3', th3)
cv2.waitKey(0)
cv2.destroyAllWindows()
