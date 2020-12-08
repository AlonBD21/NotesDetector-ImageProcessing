from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng


src = cv.imread(cv.samples.findFile("image.png"))
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))
# Detect edges using Canny
canny_output_lines = cv.Canny(src_gray, 100, 200)
canny_output = canny_output_lines.copy()
# Find contours
lines = cv.HoughLinesP(canny_output, 1, np.pi / 180, 40, minLineLength=50)
# print(lines)
for d in lines:
    x1, y1, x2, y2 = d[0][0], d[0][1], d[0][2], d[0][3]
    cv.line(canny_output_lines, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv.line(src, (x1, y1), (x2, y2), (255, 0, 0), 1)
cv.imshow("C", canny_output_lines)
cv.waitKey()
contours_lines, hierarchy_lines = cv.findContours(canny_output_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for i in range(len(contours_lines)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv.drawContours(src, contours_lines, i, color, 2, cv.LINE_8, hierarchy_lines, 0)
contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
mask_base = np.zeros(src_gray.shape, dtype="uint8")
for contour in contours:
    i = 0
    mask = mask_base.copy()
    mask = cv.drawContours(mask, [contour], -1, (255))
    mask = cv.fillPoly(mask, [contour], 255)
    mean = cv.mean(src_gray, mask=mask)
    if mean[0] > 100:
        string = "half"
        found = cv.putText(src, string, (contour[0][0][0] - 70, contour[0][0][1]),
                           cv.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
    else:
        for contour_line in contours_lines:
            if np.all(np.in1d(contour_line, contour)):
                i += 1
        print(i)
        if i == 1:
            string = "quarter"
        else:
            string = "eighth"
        found = cv.putText(src, string, (contour[0][0][0] - 70, contour[0][0][1]),
                           cv.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))

cv.imshow("Found", found)
cv.waitKey()



#cv.imshow('houghlines3.jpg', canny_output)
#cv.waitKey()
contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]
hierarchy = hierarchy[np.argsort(hierarchy[:, 3])]
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
        string = "None"
    found = cv.putText(src, str(i) + ':' + str(cv.contourArea(contour)), (contour[0][0][0] - 70, contour[0][0][1]),
                       cv.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))

#cv.imshow("Masks", masks_all)
#cv.waitKey()
