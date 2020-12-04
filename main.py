from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='Path to input image.', default='image.png')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))
# Detect edges using Canny
canny_output = cv.Canny(src_gray, 100, 100 * 2)
# Find contours

# Manipulate Source
contours, hierarchy = cv.findContours(canny_output, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]

hierarchy = hierarchy[np.argsort(hierarchy[:, 3])]
print(len(contours))

for i,contour in enumerate(contours):
    found = cv.putText(src,str(i),(contour[0][0][0]-30,contour[0][0][1]),cv.FONT_HERSHEY_TRIPLEX,1,(255,0,0))


cv.imshow("Found",found)


cv.waitKey()
