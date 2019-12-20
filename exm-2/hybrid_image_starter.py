from align_image_code import align_images
from operations import hybridImageOp, pyramidsOp, PyramidMode
import matplotlib.pyplot as plt
from utils import *
import cv2

# First load images

# high sf
im1 = plt.imread('images/DerekPicture.jpg')/255.

# low sf
im2 = plt.imread('images/nutmeg.jpg')/255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = 20
sigma2 = 21
hybrid = hybridImageOp(im1_aligned, im2_aligned, sigma1, sigma2)

# printImage("save.jpg", hybrid, False)

cv2.imwrite("save.jpg",hybrid)
# plt.show

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
# stack = pyramidsOp(im1_aligned, N, 4, mode=PyramidMode.Laplacian)
stack1 = pyramidsOp(hybrid, N, 1, mode=PyramidMode.Laplacian)
stack2 = pyramidsOp(hybrid, N, 1, mode=PyramidMode.Gaussian)

for i in range(len(stack1)):
    cv2.imwrite("laplaPyr_"+str(i)+".bmp", stack1[i])
    cv2.imwrite("gaussPyr_" + str(i) + ".bmp", stack2[i])