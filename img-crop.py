import tifffile
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

CENTER_PIXEL_X = 550
CENTER_PIXEL_Y = 440
RADIUS = 64

filename = sys.argv[1]

im = tifffile.imread(str(filename))

xmin = CENTER_PIXEL_X - RADIUS + 1
xmax = CENTER_PIXEL_X + RADIUS
ymin = CENTER_PIXEL_Y - RADIUS + 1
ymax = CENTER_PIXEL_Y + RADIUS

out = im[ : , xmin : xmax + 1 , ymin : ymax + 1]

tifffile.imwrite("cropped.tif", out)
