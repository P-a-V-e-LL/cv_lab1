import cv2
from timeit import default_timer as timer
import numpy as np
import statistics
from tqdm import tqdm
from numba import jit
import matplotlib.pyplot as plt

img1 = cv2.imread("./image.jpg",0)
kernel_sizes = [i for i in range(16) if i % 2 == 1]
print(kernel_sizes)
res = []
res2 = []
res3 = []

def gen_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    return g

def mgrd(x1, y1,x2, y2):
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
    return b, a

def gaussian_filter(image, k_size, sigma):
    height, width = image.shape[0], image.shape[1]
    # dst image height and width
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    # im2col, turn the k_size*k_size np.pixels into a row and np.vstack all rows
    image_array = np.zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i in range(dst_height):
        for j in range(dst_width):
            window = np.ravel(image[i : i + k_size, j : j + k_size])
            image_array[row, :] = window
            row += 1

    #  turn the kernel into shape(k*k, 1)
    gaussian_kernel = gen_gaussian_kernel(k_size, sigma)
    filter_array = np.ravel(gaussian_kernel)

    dst = np.dot(image_array, filter_array).reshape(dst_height, dst_width).astype(np.uint8)

    return dst

@jit(nopython=True,fastmath=True)
def gaussian_filter_numba(image, k_size, sigma):
    height, width = image.shape[0], image.shape[1]
    # dst image height and width
    dst_height = height - k_size + 1
    dst_width = width - k_size + 1

    # im2col, turn the k_size*k_size np.pixels into a row and np.vstack all rows
    image_array = np.zeros((dst_height * dst_width, k_size * k_size))
    row = 0
    for i in range(dst_height):
        for j in range(dst_width):
            window = np.ravel(image[i : i + k_size, j : j + k_size])
            image_array[row, :] = window
            row += 1

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

    # reshape and get the dst image
    dst = np.dot(image_array, filter_array).reshape(dst_height, dst_width).astype(np.uint8)

    return dst

for _ in range(5):
    gaussian_filter_numba(img1.copy(),3, sigma=1.0)

for ks in tqdm(kernel_sizes):
    temp_res = []
    temp_res2 = []
    temp_res3 = []

    for _ in range(50):
        start = timer()
        cv2.GaussianBlur(img1.copy(),(ks, ks), cv2.BORDER_DEFAULT)
        end = timer()
        elapsed_time = 1000*(end - start)
        temp_res.append(elapsed_time)
    res.append((ks,statistics.mean(temp_res)))

    for _ in range(50):
        start = timer()
        gaussian_filter(img1.copy(),ks, sigma=1.0)
        end = timer()
        elapsed_time = 1000*(end - start)
        temp_res2.append(elapsed_time)
    res2.append((ks,statistics.mean(temp_res2)))

    for _ in range(50):
        start = timer()
        gaussian_filter_numba(img1.copy(),ks, sigma=1.0)
        end = timer()
        elapsed_time = 1000*(end - start)
        temp_res3.append(elapsed_time)
    res3.append((ks,statistics.mean(temp_res3)))

plt.plot(res, label="opencv-python")
plt.plot(res2, label="numpy + loops")
plt.plot(res3, label="numpy + loops + numba")
plt.xlabel("kernel")
plt.ylabel("time, ms")
plt.legend()
plt.savefig("result.jpg")
plt.show()
