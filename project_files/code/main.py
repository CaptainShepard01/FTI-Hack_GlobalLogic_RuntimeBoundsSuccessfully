import cv2
import numpy as np
import win32gui
from PIL import ImageGrab

from decorators import performance

cap = cv2.VideoCapture("../resources/train.mp4")

STEP = 250
diapasons = np.zeros(20, dtype="int")


def to_diapason(tup):
    (decimal, integer) = tup
    if integer == 0 and decimal <= 0.2:
        return None
    if decimal - 0 > 1e-4:
        return integer
    return integer - 1


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
    # img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.Canny(img, 50, 100)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    img = cv2.erode(img, kernel, iterations=1)

    return cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE), img


@performance
def process_frames():
    i = 0
    while True:
        # #   Dynamic Version
        # img = capture_dynamic()
        #
        # if(img == None):
        #     print("No Window Found! Please Try Again")
        #     break
        #
        # img = np.array(img)
        i += 1
        (success, img) = cap.read()
        if not success:
            print(i)
            break

        ((con, hir), img) = get_frame(img)

        # new_img = np.zeros(img.shape, dtype='uint8')
        # cv2.drawContours(new_img, con, -1, (230, 111, 148), 1)
        # cv2.imshow("Result", new_img)

        # print(len(con))

        cv2.imshow('Result', img)

        if cv2.waitKey(25) & 0xFF == ord('='):
            new_img = np.zeros(img.shape, dtype='uint8')

            # con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(new_img, con, -1, (230, 111, 148), 1)

            cv2.imshow("Result", new_img)

            cv2.waitKey(0)
            # break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    process_frames()
