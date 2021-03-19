import cv2
import numpy as np

print("Your OpenCV version is: " + cv2.__version__)

#Take image as input, find shape of image
source_img = cv2.imread('source_img.jpg')
cv2.imshow('source', source_img)
cv2.waitKey(0)

r, c, channels = source_img.shape

#Question 1a,b: Downsample image and display using imshow
def downsample(img):
    print('Enter downsample factor')
    factor = int(input())

    #Extract arrays of each color channel
    blue_array = source_img[:,:,0]
    green_array = source_img[:,:,1]
    red_array = source_img[:,:,2]

    #Splice each color array using the given downsample factor as a step
    downsample_blue = blue_array[0::factor,0::factor]
    downsample_green = green_array[0::factor,0::factor]
    downsample_red = red_array[0::factor,0::factor]

    dsampled_img = np.ones(((r//factor)+1,(c//factor), channels), dtype = np.uint8)
    dsampled_img[:,:,0] = downsample_blue
    dsampled_img[:,:,1] = downsample_green
    dsampled_img[:,:,2] = downsample_red
    return dsampled_img

dsampled_img = downsample(source_img)
cv2.imshow('downsampled_img',dsampled_img)
cv2.waitKey(0)

#Question 1c: Upsample using openCV resive function
usampled_img = cv2.resize(dsampled_img, None, fx = 10, fy = 10, interpolation = cv2.INTER_NEAREST)
cv2.imshow('upsampled_img',usampled_img)
cv2.waitKey(0)
usampled_img = cv2.resize(dsampled_img, None, fx = 10, fy = 10, interpolation = cv2.INTER_LINEAR)
cv2.imshow('upsampled_img',usampled_img)
cv2.waitKey(0)
usampled_img = cv2.resize(dsampled_img, None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
cv2.imshow('upsampled_img',usampled_img)
cv2.waitKey(0)

#Question 2a: Shift image up then to the right.
shift = 80
shift_img = cv2.imread('source_img.jpg')
shift_img = downsample(shift_img)#used in testing to see shift more clearly

for i in range(0, shift):
    shift_img = np.roll(shift_img, -1, axis=0)
    shift_img[:1,:] = 0
for i in range(0, shift):
    shift_img = np.roll(shift_img, 1, axis=1)
    shift_img[:, :1] = 0

cv2.imshow('Shifted_Img', shift_img)
cv2.waitKey(0)

#Question 2b: Take filter size as input N, and calculate NxN Gaussian mask
def create_mask(N):
    z = N//2
    gaussian_mask = np.zeros((N,N), dtype=float)
    sigma = np.sqrt(2)
    x = np.arange(0-z, z+1)
    y = np.arange(0-z, z+1)
    xv, yv = np.meshgrid(x, y)
    d = (xv*xv+yv*yv)
    gaussian_mask = np.exp(-(d/(2.0*sigma**2)))
    k = 1/gaussian_mask[0,0]

    gaussian_mask *= k
    gaussian_mask = np.ceil(gaussian_mask)
    gaussian_mask = gaussian_mask.astype(int)
    gaussian_mask = gaussian_mask/np.sum(gaussian_mask)
    return gaussian_mask

print("Enter size of Gaussian mask")
N = int(input())
kernel = create_mask(N)

def apply_filter(kernel, img):
    kr,kc = kernel.shape
    r,c, channels = img.shape

    blue_array = img[:, :, 0]
    green_array = img[:, :, 1]
    red_array = img[:, :, 2]

    blue_output = np.ones((r,c))
    green_output = np.ones((r,c))
    red_output = np.ones((r,c))

    z = kr//2
    for i in range(z, r-z):
        for j in range(z, c-z):
            blue_output[i,j] = np.sum(blue_array[i-z:i+z+1, j-z:j+z+1]*kernel)
            green_output[i, j] = np.sum(green_array[i - z:i + z + 1, j - z:j + z + 1] * kernel)
            red_output[i, j] = np.sum(red_array[i - z:i + z + 1, j - z:j + z + 1] * kernel)

    output = np.ones((r, c, channels), dtype = np.uint8)
    output[:,:,0] = blue_output
    output[:,:,1] = green_output
    output[:,:,2] = red_output
    return output

filtered_img = apply_filter(kernel, source_img)
cv2.imshow("Gaussian_Filtered", filtered_img)
cv2.waitKey(0)

#Questing 2c: Take two scales and create two Gaussian filters. Subtract them and output result.
def gaussian_difference(a,b, img):
    a_mask = create_mask(a)
    b_mask = create_mask(b)
    a_img = apply_filter(a_mask,img)
    b_img = apply_filter(b_mask,img)
    difference_img = np.subtract(a_img,b_img)
    return difference_img

print("Enter two different scales for two Gaussian masks:")
a = int(input())
b = int(input())
diff_img = gaussian_difference(a,b,source_img)
cv2.imshow("Gaussian_Difference",diff_img)
cv2.waitKey(0)


#Question 3a: Apply sobel filters on x and y axes
sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=float)
sobel_x_img = np.zeros((r,c))
for i in range(2, r-2):
    for j in range(2, c-2):
        sobel_x_img[i,j] = np.sum(source_img[i-1:i+2,j-1:j+2]*sobel_x)

sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=float)
sobel_y_img = np.zeros((r,c))
for i in range(2, r-2):
    for j in range(2, c-2):
        sobel_y_img[i,j] = np.sum(source_img[i-1:i+2,j-1:j+2]*sobel_y)

cv2.imshow("Sobel_X",sobel_x_img)
cv2.waitKey(0)
cv2.imshow("Sobel_Y",sobel_y_img)
cv2.waitKey(0)

#Question 3b,c: Calculate orientation, as well as gradient magnitude, of each pixel based on gradients from (3a).
grad_orientation = np.arctan(sobel_y_img, sobel_x_img)
cv2.imshow("Grad_Orientation",grad_orientation)
cv2.waitKey(0)

grad_magnitude = np.hypot(sobel_x_img, sobel_y_img)
cv2.imshow("Grad_Magnitude",grad_magnitude)
cv2.waitKey(0)

#Question 3d: Detect edges of image using opencv canny function
canny_edge = cv2.Canny(source_img, 100, 200)
cv2.imshow("Canny_Edge",canny_edge)
cv2.waitKey(0)

