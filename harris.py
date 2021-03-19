import cv2
import numpy as np
import math

source_img = cv2.imread(r'C:\Users\domha\PycharmProjects\comp425\Yosemite1.jpg')
source_img2 = cv2.imread(r'C:\Users\domha\PycharmProjects\comp425\Yosemite2.jpg')

def getKeyPoints(source_img):
    r, c, channel = source_img.shape
    # convert the input image into grayscale
    grey_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    # modify the data type to 32 bit float
    grey_Image = np.float32(grey_img)

    #Find gradients using sobel operator, and display with imshow
    Ix = cv2.Sobel(grey_img, cv2.CV_32F,1,0,ksize=3)
    Iy = cv2.Sobel(grey_img, cv2.CV_32F,0,1,ksize=3)
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    cv2.imshow("Ix", Ix)
    cv2.imshow("Iy", Iy)
    cv2.imshow("Ixy", Ixy)
    cv2.waitKey(0)

    Ixx = cv2.GaussianBlur(Ixx, (3,3), cv2.BORDER_DEFAULT)
    Iyy = cv2.GaussianBlur(Iyy, (3,3), cv2.BORDER_DEFAULT)
    Ixy = cv2.GaussianBlur(Ixy, (3,3), cv2.BORDER_DEFAULT)

    cv2.imshow("Ixblur", Ix)
    cv2.imshow("Iyblur", Iy)
    cv2.imshow("Ixyblur", Ixy)
    cv2.waitKey(0)

    k = 0.04
    window_size = 3
    z = window_size//2
    responses = np.zeros((r,c))

    keypoints = []

    for i in range(z, r-z):
        for j in range(z, c-z):
            #Establish window in each squared gradient space
            windowIxx = Ixx[i-z: i+z+1, j-z: j+z+1]
            windowIyy = Iyy[i - z: i + z + 1, j - z: j + z + 1]
            windowIxy = Ixy[i-z: i+z+1, j-z: j+z+1]
            #Find sum of values in each window
            sumIxx = windowIxx.sum()
            sumIyy = windowIyy.sum()
            sumIxy = windowIxy.sum()
            #Find det(M) and trace(M)
            detM = (sumIxx*sumIyy) - (sumIxy**2)
            traceM = sumIxx + sumIyy
            #Calculate response score
            responses[i,j] = detM - k * (traceM**2)

            #Turn pixels with responses above the threshold into keypoints
            if(responses[i,j] > 100000000000):
                #source_img[i,j] = (0,0,0)
                keypoint = cv2.KeyPoint(j, i, responses[i, j])
                keypoints.append(keypoint)
    return keypoints, responses

keypoints, responses = getKeyPoints(source_img)
keypoints2, responses2 = getKeyPoints(source_img2)
source_img = cv2.drawKeypoints(source_img, keypoints, source_img, color=(0,0,0))
source_img2 = cv2.drawKeypoints(source_img2, keypoints, source_img2, color=(0,0,0))
#Show response image, and show source image with keypoints added
cv2.imshow("response", responses)
cv2.waitKey(0)
cv2.imshow("drawKp", source_img)
cv2.imshow("drawKp2", source_img2)
cv2.waitKey(0)


#Load new images
source_img1 = cv2.imread(r'C:\Users\domha\PycharmProjects\comp425\Yosemite1.jpg')
source_img2 = cv2.imread(r'C:\Users\domha\PycharmProjects\comp425\Yosemite2.jpg')

#use sift object to detect keypoints and calculate their descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(source_img1,None)
kp2, des2 = sift.detectAndCompute(source_img2,None)

#create lists to hold keypoints at each stage of selection
matched_kp1 = []
matched_kp2 = []
unambiguous_matched_kp1 = []
unambiguous_matched_kp2 = []

#create array to hold ssd values of best match and second best match features
match_scores = np.zeros((len(des1),2))

unambiguous_des = []
descriptor_size = 128

for i in range(len(des1)):
    best_score = 1000000000
    second_best_score = 10000000000
    matched_kp1.append(kp1[i])
    for j in range(len(des2)):
        ssd = 0
        #calculate ssd between features
        feature_difference = np.subtract(des1[i],des2[j])
        squared_difference = feature_difference**2
        ssd += sum(squared_difference)
        #if ssd is new best/second best, add it to match_scores
        if(ssd<best_score):
            best_score = ssd
            match_scores[i][0] = best_score
            best_kp2 = kp2[j]
        elif(ssd<second_best_score):
            second_best_score = ssd
            match_scores[i][1] = second_best_score

    matched_kp2.append(best_kp2)

#Calculate ratio between ssd for each keypoint
for i in range(len(matched_kp1)):
    ratio = match_scores[i][0]/match_scores[i][1]
    #If ratio is far away from 1, the second best descriptor is dissimilar to the best descriptor
    #Therefore, keep the keypoint as it is not ambiguous.
    if(ratio<0.009):
        unambiguous_matched_kp1.append(matched_kp1[i])
        unambiguous_matched_kp2.append(matched_kp2[i])

#draw keypoints on source images, and display with imshow
source_img1=cv2.drawKeypoints(source_img1,kp1,source_img1)
source_img2=cv2.drawKeypoints(source_img2,kp2,source_img2)
cv2.imwrite('sift_keypoints.jpg',source_img1)
cv2.imwrite('sift_keypoints.jpg',source_img2)

cv2.imshow("Source1", source_img1)
cv2.imshow("Source2", source_img2)
cv2.waitKey(0)

#draw matches on the images using drawMatches()
matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(len(unambiguous_matched_kp1))]
matched_img = cv2.drawMatches(source_img1, unambiguous_matched_kp1, source_img2, unambiguous_matched_kp2, matches, None)
cv2.imshow("matchy", matched_img)
cv2.waitKey(0)