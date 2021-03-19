The section of code for each question is marked by comments.
Instructions are as follows:

>>All cv.imshow functions are followed by cv.waitKey(0). Proceed to the next question/piece of code
  by pressing spacebar.

1a) Function takes the factor by which the image will be scaled down,
    and returns the scaled down image. The user is prompted to enter this 
    factor manually. As a result, restart process to test scaling with factor 2, 4, 8 and 16.

1b) After selecting downsample factor of 16, proceed to (1b) which upsamples the image
    from (1a) by a factor of 10 using cv.resize.

2b) User is prompted to enter kernel size N, and an NxN gaussian kernel is calculated then returned.
    Kernel is applied to source image and displayed with cv.imshow.

2c) User is prompted to enter two different kernel sizes, a and b.

