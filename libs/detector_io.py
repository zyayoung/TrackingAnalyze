#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import os
from .yolo import YOLO
import numpy as np
import cv2

myYOLO = YOLO()

class YoloDetector:

    def __init__(self, filepath, image, classListPath=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath

        imgSize = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]

        self.imgSize = imgSize

        self.verified = False
        # try:
        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        im = np.concatenate([np.expand_dims(im, -1)]*3, axis=-1)
        _, self.dets = myYOLO.detect_image(im)
        self.parseDets()
        # except:
            # pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, xmin, ymin, xmax, ymax, si, co, difficult):

        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), ((xmin+xmax)/2+2*co, (ymin+ymax)/2+2*si)]
        self.shapes.append((label, points, None, None, difficult))

    def parseDets(self):
        for det in self.dets:
            score = det[0]
            box = det[[2,1,4,3,5,6]]
            xmin, ymin, xmax, ymax, si, co = box

            # Caveat: difficult flag is discarded when saved as yolo format.
            self.addShape("mouse", xmin, ymin, xmax, ymax, si, co, False)
