import numpy as np
import tensorflow as tf
import cv2

def make_image():
    white_img = np.ones((144,72,3), dtype=np.uint8) * 0
    cv2.imshow('white', white_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.imwrite('./test.jpg', img=white_img)

if __name__ == '__main__':
    make_image()