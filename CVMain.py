import cv2 as cv
import numpy as np


class ShowRectangle:
    def __init__(self, haystack, needle, threshold):
        self.threshold = threshold
        self.haystack = haystack
        self.needle = needle

        self.needle = cv.imread(self.needle, cv.IMREAD_UNCHANGED)
        self.haystack = cv.imread(self.haystack, cv.IMREAD_UNCHANGED)

        self.result = cv.matchTemplate(self.haystack, self.needle, cv.TM_CCOEFF_NORMED)
        self.min_val, self.max_val, self.min_loc, self.max_loc = cv.minMaxLoc(self.result)

        self.needle_w = self.needle.shape[1]
        self.needle_h = self.needle.shape[0]

        self.top_left = self.max_loc

    def show(self):

        self.locations = np.where(self.result >= self.threshold)
        self.locations = list(zip(*self.locations[::-1]))
        rectangles = []
        for loc in self.locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            rectangles.append(rect)
            rectangles.append(rect)

        if len(rectangles):
            print(rectangles)
            rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)

            print('Needle found.')
            print('Best match top left position: %s' % str(self.max_loc))
            print('Best match top left position: %s' % str(self.max_val))

            for (x, y, w, h) in rectangles:
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv.rectangle(self.haystack, top_left, bottom_right, color=(0, 255, 0), thickness=1, lineType=cv.LINE_4)
            cv.imshow('result', self.haystack)
            cv.waitKey()

        else:
            print('Needle not fount.')




haystack_png = 'snipped/haystack.png'
needle_png = 'snipped/pie_dish.png'
test = ShowRectangle(haystack_png, needle_png, 0.9)
test.show()

