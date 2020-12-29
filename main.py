import cv2 as cv
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

# drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
"""
קבועים ביחס לתמונה
אוריאנטציה של תו

"""
MEAN_THRESH = 90
CONTOUR_LENGTH_THRESH = 20
RECTANGLE_RATIO = 0.9
Y_AXIS_ERROR = 0.1
RESIZE_FACTOR = 1.2
MINIMAL_ARC_LENGTH = 0
MINIMAL_AREA = 0
NOTE_ERROR = 0.05
NOTES = ['E', 'D', 'C', 'B', 'A', 'G', 'F', 'E']


def load_image_gray(address):
    src = cv.imread(cv.samples.findFile(address))
    src = cv.resize(src, (int(src.shape[1] * RESIZE_FACTOR), int(src.shape[0] * RESIZE_FACTOR)))
    return src


def initial_manipulation(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    # img = cv.erode(img,kernel)
    return img


def canny_algorithm(img):
    canny = cv.Canny(img, 100, 200)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    canny_temp = cv.dilate(canny, kernel)
    return canny_temp


def find_contours(canny):
    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv.arcLength(contour, True) > MINIMAL_ARC_LENGTH
            and cv.contourArea(contour) > MINIMAL_AREA]


def contour_center_of_mass(contour):
    kPcnt = len(contour)
    x = 0
    y = 0
    for kp in contour:
        x = x + kp[0][0]
        y = y + kp[0][1]
    return x / kPcnt, y / kPcnt


def distance_error_y(a, b, h):
    return np.abs(a - b) / h


def find_weight(image, contour):
    mass_x, mass_y = contour_center_of_mass(contour)
    x, y, w, h = cv.boundingRect(contour)
    center_x, center_y = x + w / 2, y + h / 2
    error_y = distance_error_y(mass_y, center_y, h)
    rectangle_ratio = w / h
    mean = mean_of_contour(image, contour)[0]
    if mean > MEAN_THRESH:
        if rectangle_ratio > RECTANGLE_RATIO:
            return "whole", mean
        else:
            return "half", mean
    else:
        if error_y < Y_AXIS_ERROR:
            return "eighth", mean
        if rectangle_ratio > RECTANGLE_RATIO:
            return "eighth", mean
        else:
            return "quarter", mean


def color(string):
    if string == "whole":
        return 0, 0, 0
    elif string == "half":
        return 0, 0, 255
    elif string == "quarter":
        return 255, 0, 0
    else:
        return 0, 255, 0


def remove_lines(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25, 1))
    detected_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv.drawContours(image, [c], -1, (255, 255, 255), 3)

    # Repair image
    repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 6))
    result = 255 - cv.morphologyEx(255 - image, cv.MORPH_CLOSE, repair_kernel, iterations=2)
    result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    threshed = cv.threshold(result_gray, 220, 255, cv.THRESH_BINARY)[1]
    return threshed


def detect_lines_index(image):
    lines = [index for index in range(len(image)) if np.average(image[index]) <= 30]
    print(len(lines) / len([index for index in lines if index - 1 not in lines]))
    return [index for index in lines if index - 1 not in lines]


def find_maximum(contour):
    y = contour[:, 0, 1]
    x = find_local_minima(y)
    return [(contour[height][0][0], contour[height][0][1]) for height in x]


def find_local_minima(arr):
    local_maxima = []
    for i in range(len(arr)):
        temp = True
        for j in arr[max(0, i - 15):min(i + 15, len(arr) - 1)]:
            if arr[i] > j:
                temp = False
        if temp:
            local_maxima.append(i)

    return [index for index in local_maxima if index - 1 not in local_maxima][: -1]


def get_note(lines_arr, top_location):
    y = top_location[1]
    average_dist = np.average(np.diff(np.array(lines_arr)))
    first_line = lines_arr[0]
    jumps = y - first_line
    return NOTES[int((jumps / average_dist * 2 + 0.3) % 7)]


def show(image):
    cv.imshow("l", image)
    cv.waitKey()


def num_of_contour(image):
    contour, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return len(contour)


def main():
    src = load_image_gray("im.jpeg")
    original = src.copy()
    show(original)
    cv.waitKey()
    lines = detect_lines_index(initial_manipulation(src))
    no_lines_src = remove_lines(src)
    canny = canny_algorithm(no_lines_src)
    ext_contours = find_contours(canny)
    for contour in ext_contours:
        max = find_maximum(contour)
        weight, mean = find_weight(initial_manipulation(original), contour)
        r, g, b = color(weight)
        cv.fillPoly(original, [contour], (r, g, b))
        for min in max:
            max_x = min[0]
            max_y = min[1]
            note = get_note(lines, (max_x, max_y))
            cv.putText(original, str(mean)[:4], (max_x, max_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    cv.imshow("Found", original)
    cv.waitKey()


def mean_of_contour(src, contour):
    mask = np.zeros(src.shape, dtype="uint8")
    cv.fillPoly(mask, [contour], 255)
    return cv.mean(src, mask=mask)


if __name__ == "__main__":
    main()
