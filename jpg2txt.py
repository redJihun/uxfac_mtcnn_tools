import numpy as np
import cv2

img = cv2.imread('/home/hong/Documents/python-mtcnn/testImage.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(np.shape(img))
img = np.swapaxes(img, 0, 2)
# pixel_list = np.reshape(img, (3, 10368))
# print(np.shape(pixel_list))
print(np.shape(img))
with open('testImage.txt', 'w') as file:
    # for line in pixel_list:
    #     for item in line:
    #         file.write('{} '.format(item))
    #     file.write('\n')
    for i in range(72):
        for j in range(36):
            file.write('{}\t'.format(img[0][j*2][i*2+1]))
        file.write('\n')


