import tifffile
import os
import pathlib
import sys

import numpy as np

CENTER_PIXEL_X = 550
CENTER_PIXEL_Y = 440
RADIUS = 64

targetDirectory = pathlib.Path(sys.argv[1])
actuatorList = [x for x in targetDirectory.iterdir() if x.is_dir()]
actuatorPath = actuatorList[0]

calibrationImageSets = list(actuatorPath.glob('*.tif*'))
    
for i, filename in enumerate(calibrationImageSets):

    im = tifffile.imread(str(filename))

    xmin = CENTER_PIXEL_X - RADIUS + 1
    xmax = CENTER_PIXEL_X + RADIUS
    ymin = CENTER_PIXEL_Y - RADIUS + 1
    ymax = CENTER_PIXEL_Y + RADIUS

    out = im[ : , xmin : xmax + 1 , ymin : ymax + 1]

    tifffile.imwrite(f"cropped_{i}.tif", out)
