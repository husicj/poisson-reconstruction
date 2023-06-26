
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

def crop_images(sourceDirectory, targetDirectory, actuatorList):
    for actuator in actuatorList:
        calibrationImageSets = list(actuator.glob('*.tif*'))
        actuatorPathName = actuator.parts[-1]
        actuatorTargetDirectory = os.path.join(str(targetDirectory), actuatorPathName)
        if not os.path.isdir(actuatorTargetDirectory):
            os.mkdir(actuatorTargetDirectory)

        for file in calibrationImageSets:
            im = tifffile.imread(str(file))

            stackCenter = im[STACK_CENTER_INDEX, :, :]
            shape = stackCenter.shape
            centerPixelX, centerPixelY = find_brightest_region(stackCenter)
            centerPixelX, centerPixelY = pad_edge(shape, centerPixelX, centerPixelY)
            print(f"Using center pixel ({centerPixelX}, {centerPixelY}) for cropping image at {file}.")

            xmin = centerPixelX - RADIUS + 1
            xmax = centerPixelX + RADIUS
            ymin = centerPixelY - RADIUS + 1
            ymax = centerPixelY + RADIUS

            out = im[ : , xmin : xmax + 1 , ymin : ymax + 1]

            writePath = os.path.join(actuatorTargetDirectory, "cropped_" + file.parts[-1])
            tifffile.imwrite(writePath, out)


def find_brightest_region(image):
    blurredImage = scipy.ndimage.gaussian_filter(image, sigma=7) # blurred to find brightest region over noise
    X, Y = np.where(blurredImage == blurredImage.max())
    return find_int_mean(X), find_int_mean(Y)

def find_int_mean(array):
    mean = np.sum(array) / array.size
    return int(mean)

def pad_edge(shape, x, y):
    retX = x
    retY = y

    if (x - RADIUS + 1) < 0:
        retX = RADIUS - 1
    if (y - RADIUS + 1) < 0:
        retY = RADIUS - 1
    if (x + RADIUS) > shape[0]:
        retX = shape[0] - RADIUS
    if (y + RADIUS) > shape[1]:
        retY = shape[1] - RADIUS

    return retX, retY

def setup():
    try:
        global sourceDirectory
        sourceDirectory = pathlib.Path(sys.argv[1])
        global targetDirectory
        targetDirectory = pathlib.Path(sys.argv[2])
        if not os.path.isdir(targetDirectory):
            os.mkdir(targetDirectory)
    except IndexError:
        print("usage: \"python img-crop.py {source_dir} {target_dir}\"")
        exit(1)

    global actuatorList
    actuatorList = [x for x in sourceDirectory.iterdir() if x.is_dir()]

#####
##### CROP IMAGES
#####

if __name__ == "__main__":
    setup()

    crop_images(sourceDirectory = sourceDirectory,
                targetDirectory = targetDirectory,
                actuatorList = actuatorList)
