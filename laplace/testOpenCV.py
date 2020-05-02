import cv2
import numpy as np

img = cv2.imread('resources/face.png')

kernel = np.array([[0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lap = cv2.filter2D(gray, -1, kernel)

cv2.imshow('asd', lap)

cv2.waitKey()


