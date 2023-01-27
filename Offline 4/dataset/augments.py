import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

def rotate(image, angleFrom=-10, angleTo=10):
    '''
    angles in degrees
    '''
    angle = np.random.randint(angleFrom, angleTo)
    return ndimage.rotate(image, angle)


def blur(image):
    sigma = np.random.randint(3, 6)
    return ndimage.gaussian_filter(image, sigma=sigma)


def get_number_bb(image):
    # convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (_, binary) = cv2.threshold(gray, 255//2, 255, cv2.THRESH_OTSU)
    binary = 255-binary

    # apply erosion + dilation to remove noise
    kernel = np.ones((5,5),np.uint8)
    img_opening = cv2.erode(cv2.dilate(binary,kernel,iterations = 1), kernel,iterations = 1)
    # plt.figure()
    # plt.imshow(img_opening, cmap='gray')

    # get bounding box
    contours, _ = cv2.findContours(img_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]
    bounding_box = cv2.boundingRect(contours)
    # print(bounding_box, image.shape)

    return image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2], :]
