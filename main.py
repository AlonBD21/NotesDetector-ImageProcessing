import cv2 as cv
import numpy as np

# drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

MEAN_THRESH = 99
CONTOUR_LENGTH_THRESH = 20
RECTANGLE_RATIO = 0.85
Y_AXIS_ERROR = 0.1
RESIZE_FACTOR = 1.2


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
    canny = cv.dilate(canny, kernel)
    return canny


def find_contours(canny):
    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv.contourArea(contour) > 50]


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


def find_weight(src, contour):
    mass_x, mass_y = contour_center_of_mass(contour)
    x, y, w, h = cv.boundingRect(contour)
    center_x, center_y = x + w / 2, y + h / 2
    error_y = distance_error_y(mass_y, center_y, h)
    rectangle_ratio = w / h
    mean = mean_of_contour(src, contour)[0]
    print(mean)
    if mean > MEAN_THRESH:
        if rectangle_ratio > RECTANGLE_RATIO:
            return "whole"
        else:
            return "half"
    else:
        if error_y < Y_AXIS_ERROR:
            return "eighth"
        else:
            return "quarter"


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
        cv.drawContours(image, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 6))
    result = 255 - cv.morphologyEx(255 - image, cv.MORPH_CLOSE, repair_kernel, iterations=1)
    result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    threshed = cv.threshold(result_gray, 220, 255, cv.THRESH_BINARY)[1]
    return threshed


def show(image):
    cv.imshow("l", image)
    cv.waitKey()


def main():
    src = load_image_gray("line.png")
    original = src.copy()
    no_lines_src = remove_lines(src)
    canny = canny_algorithm(no_lines_src)
    show(canny)
    ext_contours = find_contours(canny)
    for contour in ext_contours:
        weight = find_weight(no_lines_src, contour)
        r, g, b = color(weight)
        cv.fillPoly(original, [contour], (r, g, b))
    cv.imshow("Found", original)
    cv.waitKey()


def mean_of_contour(src, contour):
    mask = np.zeros(src.shape, dtype="uint8")
    mask = cv.fillPoly(mask, [contour], 255)
    mask = tighten_mask(mask)
    return cv.mean(src, mask=mask)


def tighten_mask(mask):
    mask = cv.erode(mask, (3, 3))
    return mask


if __name__ == "__main__":
    main()
