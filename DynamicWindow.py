import cv2
import numpy as np


def nothing(x):
    pass


img = np.zeros((320, 80, 1), np.uint8)
cv2.namedWindow('canny')

cv2.createTrackbar('Lower', 'canny', 0, 255, nothing)
cv2.createTrackbar('Upper', 'canny', 0, 255, nothing)

while 1:
    cv2.imshow('canny', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    l = cv2.getTrackbarPos('Lower', 'canny')
    u = cv2.getTrackbarPos('Upper', 'canny')


