import tifffile
import os
import pathlib
import sys

import numpy as np
import scipy.ndimage

RADIUS = 64
STACK_DEPTH = 7
STACK_CENTER_INDEX = (STACK_DEPTH - 1) // 2

#####
##### FUNCTIONS
#####

def find_brightest_region(image):
    blurredImage = scipy.ndimage.gaussian_filter(image, sigma=7) # blurred to find brightest region over noise
    X, Y = np.where(blurredImage == blurredImage.max())
    return find_int_mean(X), find_int_mean(Y)

def find_int_mean(array):
    mean = np.sum(array) / array.size
    return int(mean)

#####
##### SETUP
#####

sourceDirectory = pathlib.Path(sys.argv[1])
targetDirectory = pathlib.Path(sys.argv[2])

actuatorList = [x for x in sourceDirectory.iterdir() if x.is_dir()]
actuatorPath = actuatorList[0]

calibrationImageSets = list(actuatorPath.glob('*.tif*'))

#####
##### CROP IMAGES
#####
    
for i, filename in enumerate(calibrationImageSets):

    im = tifffile.imread(str(filename))

    stackCenter = im[STACK_CENTER_INDEX, :, :]
    center_pixel_x, center_pixel_y = find_brightest_region(stackCenter)
    print(f"Using center pixel ({center_pixel_x}, {center_pixel_y}) for cropping image at {filename}.")

    xmin = center_pixel_x - RADIUS + 1
    xmax = center_pixel_x + RADIUS
    ymin = center_pixel_y - RADIUS + 1
    ymax = center_pixel_y + RADIUS

    out = im[ : , xmin : xmax + 1 , ymin : ymax + 1]

    tifffile.imwrite(f"{targetDirectory}/cropped_{i}.tif", out)
