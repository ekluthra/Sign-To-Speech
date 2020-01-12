import cv2
import numpy as np
import math
from google.cloud import vision
import PIL
import time
import os

video = cv2.VideoCapture(0)

while True:
    ret, window = video.read()
    window = cv2.flip(window, 1)
    kernel = np.ones((3, 3), np.uint8)

    box = window[100:300, 100:300]

    cv2.rectangle(window, (100, 100), (300, 300), (0, 255, 0), 0)
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)

    # skin colors in HSV
    lower_bound_skin = np.array([0, 20, 70])
    upper_bound_skin = np.array([20, 255, 255])

    # extract skin colur imagw
    mask = cv2.inRange(hsv, lower_bound_skin, upper_bound_skin)

    # extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask, kernel, iterations=4)

    # blur the image
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # get contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # get contour of max area(hand)
    contour_of_max_area = max(contours, key=lambda x: cv2.contourArea(x))

    # approximating the contour
    epsilon = 0.0005 * cv2.arcLength(contour_of_max_area, True)
    approx = cv2.approxPolyDP(contour_of_max_area, epsilon, True)

    # make convex hull around hand
    hull = cv2.convexHull(contour_of_max_area)

    # define area of hull and area of hand
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(contour_of_max_area)

    # get percentage of area that hand does not cover in convex hull
    arearatio = ((areahull - areacnt) / areacnt) * 100

    # find the defects in convex hull with respect to hand
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    num_defects = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt = (100, 180)

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        s = (a + b + c) / 2
        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

        # distance between point and convex hull
        d = (2 * ar) / a

        # apply cosine rule here
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        # ignore angles > 90 and ignore points very close to convex hull(they
        # generally come due to noise)
        if angle <= 90 and d > 30:
            num_defects += 1
            cv2.circle(box, far, 3, [255, 0, 0], -1)

        # draw lines around hand
        cv2.line(box, start, end, [0, 255, 0], 2)

    num_defects += 1

    font = cv2.QT_FONT_NORMAL
    if num_defects == 1:
        cv2.putText(window, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    elif num_defects == 2:
        cv2.putText(window, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    elif num_defects == 3:
        cv2.putText(window, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    elif num_defects == 4:
        cv2.putText(window, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    elif num_defects == 5:
        cv2.putText(window, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('mask', mask)
    cv2.imshow('window', window)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
video.release()
