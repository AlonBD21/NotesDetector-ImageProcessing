from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng


src = cv.imread(cv.samples.findFile("image.png"))
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
copy = src_gray.copy()
src_gray = cv.blur(src_gray, (2, 2))
cv.threshold(src_gray, 220, 255, cv.THRESH_BINARY)
# Detect edges using Canny
canny_output_lines = cv.Canny(src_gray, 100, 200)
canny_output = canny_output_lines.copy()
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Find contours
lines = cv.HoughLinesP(canny_output, 1, np.pi / 180, 40, minLineLength=30)
for d in lines:
    x1, y1, x2, y2 = d[0][0], d[0][1], d[0][2], d[0][3]
    cv.line(canny_output_lines, (x1, y1), (x2, y2), (0, 0, 0), 2)
contours_lines, hierarchy_lines = cv.findContours(canny_output_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
mask_base = np.zeros(src_gray.shape, dtype="uint8")
contour_weight = []
for contour in contours:
    i = 0
    mask = mask_base.copy()
    mask = cv.drawContours(mask, [contour], -1, (255))
    mask = cv.fillPoly(mask, [contour], 255)
    mean = cv.mean(copy, mask=mask)
    if mean[0] > 70:
        string = "half"
        cv.fillPoly(src, pts=[contour], color=(0, 255, 0))
        contour_weight.append((contour, string))
    else:
        x, y, w, h = cv.boundingRect(contour)
        cropped = canny_output_lines[y:y + h, x:x + w]
        temp_contours, temp_hierarchy = cv.findContours(cropped, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        i = 0
        for con in temp_contours:
            if cv.arcLength(con, False) > 20:
             i += 1
        if i > 1:
            string = "eighth"
            cv.fillPoly(src, pts=[contour], color=(255, 0, 0))
        else:
            string = "quarter"
            cv.fillPoly(src, pts=[contour], color=(0, 0, 255))
        contour_weight.append((contour, string))
    src = cv.putText(src, string, (contour[0][0][0] - 30, contour[0][0][1]),
                       cv.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
cv.imshow("Found", src)
cv.waitKey()