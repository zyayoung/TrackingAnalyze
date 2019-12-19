#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import os
import codecs
from libs.constants import DEFAULT_ENCODING

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING

class KerasYoloWriter:

    def __init__(self, foldername, filename, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def addBndBox(self, xmin, ymin, xmax, ymax, headx, heady, name, difficult):
        xcen = (xmin + xmax) / 2
        ycen = (ymin + ymax) / 2
        headVecX = headx - xcen
        headVecY = heady - ycen
        self.boxlist.append([xmin, ymin, xmax, ymax, headVecX, headVecY])

    def save(self, classList=[], targetFile=None):

        out_file = None #Update yolo .txt

        if targetFile is None:
            out_file = open(
            self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        out_file.write(self.localImgPath + " ")
        for box in self.boxlist:
            xmin, ymin, xmax, ymax, headVecX, headVecY = box
            # print (classIndex, xcen, ycen, w, h)
            out_file.write("%d,%d,%d,%d,%d,%d,0 " % (xmin, ymin, xmax, ymax, headVecY, headVecX))

        out_file.close()



class KerasYoloReader:

    def __init__(self, filepath, classListPath=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (headx, heady)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath

        # if classListPath is None:
        #     dir_path = os.path.dirname(os.path.realpath(self.filepath))
        #     self.classListPath = os.path.join(dir_path, "classes.txt")
        # else:
        #     self.classListPath = classListPath

        # # print (filepath, self.classListPath)

        # classesFile = open(self.classListPath, 'r')
        # self.classes = classesFile.read().strip('\n').split('\n')

        # print (self.classes)

        self.verified = False
        # try:
        self.parseYoloFormat()
        # except:
            # pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, xmin, ymin, xmax, ymax, headVecX, headVecY, difficult):
        xmin =int(xmin)
        ymin =int(ymin)
        xmax =int(xmax)
        ymax =int(ymax)
        headVecX =int(headVecX)
        headVecY =int(headVecY)
        xcen = (xmin + xmax) / 2
        ycen = (ymin + ymax) / 2
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xcen + headVecX, ycen + headVecY)]
        self.shapes.append((label, points, None, None, difficult))

    def parseYoloFormat(self):
        bndBoxFile = open(self.filepath, 'r')
        for bndBox in bndBoxFile:
            _fileName, *pointList = bndBox.split(' ')
            for points in pointList:
                if not ',' in points:
                    continue
                xmin, ymin, xmax, ymax, headVecY, headVecX, *_ = points.split(',')
                self.addShape("mouse", xmin, ymin, xmax, ymax, headVecX, headVecY, False)

