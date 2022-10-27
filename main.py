import os
import numpy as np
import time
import cv2
from numba import jit
from tqdm import tqdm
import keyboard
import math

target_image = cv2.imread("./image.jpg")
kernel_size = 3

def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    return g

def gaussian_filter(image, k_size, sigma):
    height, width = image.shape[0], image.shape[1]
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    image_array = np.zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i in range(dst_height):
        for j in range(dst_width):
            window = np.ravel(image[i : i + k_size, j : j + k_size])
            image_array[row, :] = window
            row += 1

    gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
    filter_array = np.ravel(gaussian_kernel)

    dst = np.dot(image_array, filter_array).reshape(dst_height, dst_width).astype(np.uint8)

    return dst

@jit(nopython=True,fastmath=True)
def gaussian_filter_numba(image, k_size, sigma):
    height, width = image.shape[0], image.shape[1]
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    image_array = np.zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i in range(dst_height):
        for j in range(dst_width):
            window = np.ravel(image[i : i + k_size, j : j + k_size])
            image_array[row, :] = window
            row += 1

    #gaussian_kernel = gen_gaussian_kernel(k_size, sigma)

    center = k_size // 2

    x1 = 0 - center
    y1 = k_size - center
    x2 = 0 - center
    y2 = k_size - center
    a = np.arange((y1-x1)*(y2-x2)).reshape((y1-x1), (y2-x2))
    b = np.arange((y1-x1)*(y2-x2)).reshape((y1-x1), (y2-x2))
    if x1 < 0:
        d1 = abs(x1)
    if x2 < 0:
        d2 = abs(x2)
    for i in range(x2, y2):
        for j in range(x1, y1):
            a[j+d1][i+d2] = i
            b[j+d1][i+d2] = j
    x, y = b, a

    gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

    filter_array = np.ravel(gaussian_kernel)

    dst = np.dot(image_array, filter_array).reshape(dst_height, dst_width).astype(np.uint8)

    return dst

print("Press 'z' to show original image.")
print("Press 'x' to show gauss filter.")

while True:
    if keyboard.is_pressed('z'):
        print()
        print('Original Image')
        #rgb_img = cv2.cvtColor(target_image,cv2.COLOR_BGR2RGB)
        cv2.imshow(('Image'),target_image)
        cv2.waitKey(0)
        time.sleep(0.1)

    if keyboard.is_pressed('x'):
        print()
        print('Gauss Filter')

        # с помощью библиотеки cv2
        _img = cv2.GaussianBlur(target_image, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)

        # циклы
        #_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        #_img = gaussian_filter(_img, kernel_size, sigma=1.0)

        # циклы + numba
        #_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        #_img = gaussian_filter_numba(_img, kernel_size, sigma=1.0)

        _img = np.uint8(_img)
        cv2.imshow('Image',_img)
        #cv2.imwrite('gauss.jpg', _img)
        cv2.waitKey(0)
        time.sleep(0.1)
