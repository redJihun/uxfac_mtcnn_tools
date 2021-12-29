import numpy as np
import cv2

file = open('uxfac_mtcnn_keras/uxfac_train/12x12_hw_img.txt', 'r')
txt = file.read().split()
# for i in range(len(txt)):
#     txt[i] = int(txt[i]) / 255.0
txt = np.reshape(txt, (12,12))
print(txt.shape)
print(type(txt))
txt = np.array(txt, np.float32)
# for row in len(txt):
#     txt[:, row] =
# print(txt)
img = cv2.cvtColor(txt, cv2.COLOR_GRAY2BGR)
print(np.shape(img))
cv2.imwrite('txt2jpg.jpg', img)