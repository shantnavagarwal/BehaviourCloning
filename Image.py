import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def nothing(x):
    pass


def dynamic(can):
    zeroimg = np.zeros_like(can, dtype=np.uint8)
    cv2.namedWindow('canny')
    # can = cv2.GaussianBlur(can[:, :, 2], (3, 3), 0)
    cv2.createTrackbar('RLower', 'canny', 0, 255, nothing)
    cv2.createTrackbar('RUpper', 'canny', 0, 255, nothing)
    cv2.createTrackbar('GLower', 'canny', 0, 255, nothing)
    cv2.createTrackbar('GUpper', 'canny', 0, 255, nothing)
    cv2.createTrackbar('BLower', 'canny', 0, 255, nothing)
    cv2.createTrackbar('BUpper', 'canny', 0, 255, nothing)
    bcan = can[:, :, 0]
    gcan = can[:, :, 1]
    rcan = can[:, :, 2]

    while 1:
        cv2.imshow('canny', zeroimg)
        cv2.imshow('img', can)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        rl = cv2.getTrackbarPos('RLower', 'canny')
        ru = cv2.getTrackbarPos('RUpper', 'canny')
        bl = cv2.getTrackbarPos('BLower', 'canny')
        bu = cv2.getTrackbarPos('BUpper', 'canny')
        gl = cv2.getTrackbarPos('GLower', 'canny')
        gu = cv2.getTrackbarPos('GUpper', 'canny')

        zeroimg[((can[:, :, 0] > bl) & (can[:, :, 0] < bu)) | ((can[:, :, 1] > gl) & (can[:, :, 1] < gu)) |
                ((can[:, :, 2] > rl) & (can[:, :, 2] < ru))] = 250


images = np.load('OrigXTrain.npy')
img = images[2100]
img = img[60:140, :]
# plt.imshow(img)
# plt.show()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(gray, 150, 200)
# plt.imshow(canny)
# plt.show()
src = np.array([[100, 0], [0, 40], [310, 40], [220, 0]], dtype='float32')
dst = np.array([[0, 0], [0, 79], [319, 79], [319, 0]], dtype='float32')
M = cv2.getPerspectiveTransform(src, dst)
# warpimg = cv2.warpPerspective(img, M, dsize=(320, 80))
# plt.imshow(warpimg[:, :, 2], cmap='gray')
# plt.show()
# plt.imshow(warpimg)
# plt.show()
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # plt.imshow(gray, cmap='gray')
# # plt.show()
# canny = cv2.warpPerspective(canny, M, dsize=(320, 80))
# plt.imshow(canny, cmap='gray')
# plt.show()

dynamic(img)
