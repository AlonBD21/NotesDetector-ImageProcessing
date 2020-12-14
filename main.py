import cv2 as cv
import numpy as np

# drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

MEAN_THRESH = 70
CONTOUR_LENGTH_THRESH = 20


def load_image_gray(address):
    src = cv.imread(cv.samples.findFile(address))
    src = cv.resize(src, (int(src.shape[1] * 1.2), int(src.shape[0] * 1.2)))
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
    canny = cv.dilate(canny,kernel)
    return canny


def find_contours(canny):
    contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def main():
    src = load_image_gray("image.png")
    original = src.copy()
    src = initial_manipulation(src)
    src_copy = src.copy()

    canny = canny_algorithm(src)
    ext_contours = find_contours(canny)


    cv.drawContours(original, ext_contours, -1, (0, 255, 0))
    for contour in ext_contours:
        mean = mean_of_contour(src, contour)[0]  # Channel 0
        if mean > MEAN_THRESH:
            string = "half"
        else:
            string = "eighth or quarter"

        cv.putText(original, string+" "+str(mean)[:4], (contour[0][0][0] - 100, contour[0][0][1]),
                          cv.FONT_HERSHEY_DUPLEX, 0.5, (100, 200, 0))
    cv.imshow("Found", original)
    cv.imshow("canny",canny)
    cv.imshow("src",src)
    cv.waitKey()


def mean_of_contour(src, contour):
    mask = np.zeros(src.shape, dtype="uint8")
    mask = cv.fillPoly(mask, [contour], 255)
    return cv.mean(src, mask=mask)



if __name__ == "__main__":
    main()
