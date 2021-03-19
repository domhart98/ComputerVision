import cv2
import numpy as np
import math

print("Your OpenCV version is: " + cv2.__version__)

#Take image as input, find shape of image
source_img = cv2.imread(r'C:\Users\domha\PycharmProjects\comp425\hough2.png')
cv2.imshow('source', source_img)
cv2.waitKey(0)

#find width and height, then calculate diagonal(maximum) distance
r, c, channel = source_img.shape
rho_size = int(np.ceil(math.sqrt((r*r) + (c*c))))
theta_size = 180

#Double rho space to accomodate negative rho values.
hough_space = np.zeros((2*rho_size, theta_size), dtype = np.uint8)

#Blur image and get edge map
source_img = cv2.GaussianBlur(source_img, (3,3), cv2.BORDER_DEFAULT)
edge_map = cv2.Canny(source_img, 255, 1, 0)

grad_x = cv2.Sobel(edge_map,cv2.CV_64F,1,0,ksize=3)
grad_y = cv2.Sobel(edge_map,cv2.CV_64F,0,1,ksize=3)

grad_orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
grad_magnitude = cv2.magnitude(grad_x, grad_y)

#get indices of edge pixels and put in arrays
y_index, x_index = np.nonzero(edge_map)

for i in range(len(x_index)):
    x = x_index[i]
    y = y_index[i]

    for theta in range(theta_size):
        rho = rho_size + int(np.ceil(x * np.cos(math.radians(theta)) + y * np.sin(math.radians(theta))))
        hough_space[rho][theta] += 1

cv2.imshow("Hough_Space", hough_space)
cv2.waitKey(0)

#Threshold accumulator values
edge_rho, edge_theta = np.where(hough_space > 70)

#find endpoints of the line, then draw line on original image
for i in range(len(edge_rho)):
    a = np.cos(math.radians(edge_theta[i]))
    b = np.sin(math.radians(edge_theta[i]))
    x0 = a * (edge_rho[i] -rho_size)
    y0 = b * (edge_rho[i] -rho_size)
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(source_img, (x1, y1), (x2, y2), (255, 255, 255))

cv2.imshow('Output_Img', source_img)
cv2.waitKey(0)

