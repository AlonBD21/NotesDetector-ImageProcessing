# Standard imports
import cv2
import numpy as np;

# Read image
im = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector()

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(im, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
# Show keypoints
cv2.imwrite("Keypoints.jpg", blobs)
cv2.waitKey(0)