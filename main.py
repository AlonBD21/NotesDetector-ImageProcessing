from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

def is_eighth(mask):
    for line in mask:
        found_black = False
        found_white_after_black = False
        for pixel in line:
            if pixel < 20:
                found_black = True
            if found_black and pixel > 150:
                print("found")
                found_white_after_black = True
            if found_white_after_black and pixel < 20:
                return True

    return False

src = cv.imread(cv.samples.findFile("image.png"))
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))
# Detect edges using Canny
canny_output = cv.Canny(src_gray, 100, 200)
# Find contours

contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]
hierarchy = hierarchy[np.argsort(hierarchy[:, 3])]
print(len(contours))

mask_base = np.zeros(src_gray.shape, dtype="uint8")
masks_all = mask_base.copy()
# Review And Print
for i, contour in enumerate(contours):
    mask = mask_base.copy()
    mask = cv.drawContours(mask, [contour], -1, (255))
    mask = cv.fillPoly(mask, [contour], 255)
    masks_all = cv.fillPoly(masks_all, [contour], 255)
    mean = cv.mean(src_gray, mask=mask)
    if mean[0] > 100:
        string = "half"
    else:
        if is_eighth(mask):
            string = "eighth"
        else:
            string = "quarter"
    found = cv.putText(src, str(i) + ":" + string, (contour[0][0][0] - 30, contour[0][0][1]),
                       cv.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
    is_eighth(mask)
    cv.waitKey()

cv.imshow("Canny", canny_output)
cv.imshow("Found", found)
cv.waitKey()
cv.imshow("Masks",masks_all)
cv.waitKey()


