import cv2
import os
import glob
import numpy as np
import open3d as o3d
import dbow

class config:
    def __init__(self):
        self.dataset = config.dataset()
        self.sparse = config.sparse()
        self.opt = config.opt()

    class dataset:
        def __init__(self):
            self.path = os.getcwd() + '/dataset/'
            self.intscale = 1.2
            self.imresize = 1
            self.imrotate = 0
            self.minz = 0
            self.maxz = 3
            self.display = False

    class sparse:
        def __init__(self):
            self.threshold = 0.6
            self.feature = 'AKAZE'
            self.display = True
            self.detector = cv2.AKAZE_create(threshold=0.0001)
            self.matcher = cv2.BFMatcher()

    class opt:
        def __init__(self):
            self.threshold = 0.9