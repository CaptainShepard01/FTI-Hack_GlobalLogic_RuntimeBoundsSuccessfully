from math import modf

import cv2
import numpy as np
import win32gui
from PIL import ImageGrab

from utils import *

cap = cv2.VideoCapture("../resources/test.mp4")

STEP = 250
EPS = 1e-4


def to_diapason(particle_tuple):
    (decimal, integer) = particle_tuple
    if integer >= 20 or integer == 0 and decimal <= 0.2:
        return None
    if decimal > EPS:
        return integer
    return integer - 1


def count_parts(contours, hierarchy):
    diapasons = np.zeros(20, dtype="int")
    # temp_arr = []
    for i in range(len(contours)):
        if hierarchy[0][i][2] != -1:
            continue
        separated_values = explode_xy(contours[i])
        area_of_con = shoelace_area(separated_values[0], separated_values[1])
        # temp_arr.append(area_of_con)
        if (value := to_diapason(modf(area_of_con / STEP))) is not None:
            diapasons[int(value)] += 1

        # print(sorted(temp_arr))
    return diapasons


def capture_dynamic():
    toplist, winlist = [], []

    def enum_cb(hwnd, results):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(enum_cb, toplist)

    wnd = [(hwnd, title) for hwnd, title in winlist if 'firefox' in title.lower()]

    if wnd:
        wnd = wnd[0]
        hwnd = wnd[0]

        bbox = win32gui.GetWindowRect(hwnd)
        img = ImageGrab.grab(bbox)
        return img
    else:
        return None


def get_frame(img):
    img = cv2.GaussianBlur(img, (1, 1), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.Canny(img, 20, 20)

    # kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, np.ones((4, 4), np.uint8), iterations=1)
    #
    img = cv2.erode(img, np.ones((4, 4), np.uint8), iterations=1)

    return cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE), img


@performance
def process_frames():
    # i = 0
    number_of_particles = []
    while True:
        # #   Dynamic Version
        # img = capture_dynamic()
        #
        # if(img == None):
        #     print("No Window Found! Please Try Again")
        #     break
        #
        # img = np.array(img)

        # i += 1
        (success, img) = cap.read()
        if not success:
            # print(i)
            break

        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255, 255, 255), thickness=2)
        ((con, hir), img) = get_frame(img)
        # cv2.imshow('Result', img)
        number_of_particles.append(count_parts(con, hir))

        # if cv2.waitKey(25) & 0xFF == ord('='):
        #     for i in range(len(con)):
        #         separated_values = explode_xy(con[i])
        #         if hir[0][i][2] != -1:
        #             continue
        #         area_of_con = cv2.contourArea(con[i], True)
        #         if area_of_con <= 50:
        #             area_of_con = shoelace_area(separated_values[0], separated_values[1])
        #
        #         if 50 < area_of_con <= 5000:
        #             ellipse = cv2.fitEllipse(con[i])
        #             cv2.ellipse(img, ellipse, (112, 112, 112), thickness=2)
        #
        #     # print(count_parts(con))
        #
        #     cv2.imshow("Result", img)
        #     cv2.waitKey(0)

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

    print("Test[54]: ", number_of_particles[54])
    print("Test[101]: ", number_of_particles[101])
    print("Test[256]: ", number_of_particles[256])


if __name__ == "__main__":
    process_frames()
