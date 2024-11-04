import cv2
import numpy as np
import imutils

from src.ai_engine.model.corner_detector import CornerDetector
from src.ai_engine.model.ocr_decode import OCRDecoder


def crop_transform(pts, img):
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points -- the top-right
    # will have the minimum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect*1.3
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view

    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp


def process_segment(img, thresh):
    '''
        img: numpy shape (size_img, size_img)
        return numpy shape (4,2)
    '''

    img = np.uint8((img > 0.6) * 255)
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, thresh * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        # print(approx)
        if len(approx) == 4:
            screenCnt = approx
            sc = screenCnt.reshape((4, 2))
            return sc

    return np.array([])



class Recognizer:
    def __init__(self, corner_path: str = None, ocr_path: str = None):
        self.corner_path = corner_path
        self.ocr_path = ocr_path

        self.load_model()

    def load_model(self):
        self.corner = CornerDetector(model_path=self.corner_path)
        self.ocr = OCRDecoder(model_path=self.ocr_path)

    def get_result(self, image, cls=1):
        if int(cls) == 0:
            shape = (128, 32)
        elif int(cls) == 1:
            shape = (128, 64)
        else:
            return ""

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))

        image = image.copy()
        segs = self.corner.predict(image)
        res_seg = [process_segment(segs[i], 0.07) *
                   4 for i in range(segs.shape[0])]
        if not res_seg[0].size:
            return ""
        res_seg = [crop_transform(res_seg[0], image)]
        res_seg = cv2.resize(res_seg[0], shape)

        if int(cls) == 0:
            out = self.ocr.predict(res_seg)
            out = np.mean(out, axis=0, keepdims=True)
            out = self.ocr.decode_batch(out)[0]

        elif int(cls) == 1:
            seg1 = np.array(res_seg)[:32, :]
            out1 = self.ocr.predict(seg1)
            out1 = self.ocr.decode_batch(out1)[0]

            seg2 = np.array(res_seg)[32:, :]
            out2 = self.ocr.predict(seg2)
            out2 = self.ocr.decode_batch(out2)[0]
            out = out1 + " " + out2

        return out
