import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# Read the original image
img = cv2.imread('simple_test.jpg') 
# Display original image
# cv2.imshow('Original', img)
# cv2.waitKey(0)
 
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
# Sobel Edge Detection
Gx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # Sobel Edge Detection on the X axis
Gy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis
Gxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3) # Combined X and Y Sobel Edge Detection

# Gradient magnitude
G = np.sqrt(Gx**2 + Gy**2)

# # Normalize to range 0-255
# # Gx = np.uint8(255 * np.abs(Gx) / np.max(Gx))
# # Gy = np.uint8(255 * np.abs(Gy) / np.max(Gy))
G = np.uint8(255 * G / np.max(G))

# plt.imshow(G, cmap='gray')
# plt.title('Sobel Edge Detection')
# plt.show()

# Display Sobel Edge Detection Images
# cv2.imshow('Sobel X', Gx)
# cv2.waitKey(0)
# cv2.imshow('Sobel Y', Gy)
# cv2.waitKey(0)
# cv2.imshow('Sobel X Y using Sobel() function', Gxy)
# cv2.waitKey(0)
cv2.imshow('G using Sobel() function', G)
cv2.waitKey(0)
