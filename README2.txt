COMP425 ASSIGNMENT 2
Dominic Hart
40068282

1) hough.py implements Hough transform on an image, shows the hough space of that given image,
   and draws the detected lines onto the image.

2) harris.py implements Harris corner detection function on an image. It finds the gradients on the x
   and y axes of the image and displays Ix, Iy and Ix*Iy. It also displays the response calculated for 
   each pixel, and displays the image with keypoints drawn in.

3) harris.py also loads 2 images, then finds the keypoints and their descriptors using a SIFT object.
   The descriptors are matched using the ratio algorithm. DrawMatches is then used to draw the matches onto,
   the original images and they are displayed using imshow.

To run both hough.py and harris.py: open, change the path of the source images, run, when waitKey(0) is encountered
press spacebar to continue.