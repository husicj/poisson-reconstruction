import cv2
import os
import sys

import numpy as np


CENTER_PIXEL_X = 550
CENTER_PIXEL_Y = 440
RADIUS = 64

filename = sys.argv[1]

im = cv2.imread(str(filename))
print(im.shape)

xmin = CENTER_PIXEL_X - RADIUS + 1
xmax = CENTER_PIXEL_X + RADIUS
ymin = CENTER_PIXEL_Y - RADIUS + 1
ymax = CENTER_PIXEL_Y + RADIUS

out = im[ xmin : xmax + 1 , ymin : ymax + 1, : ]
cv2.imwrite("cropped.tif", out)
