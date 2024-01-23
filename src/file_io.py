try:
    import imageio.v3 as imageio
except (ImportError, ModuleNotFoundError):
    import imageio
 
import glob
import os
import pathlib
import re
import sys

import img_crop

import numpy as np

def scan_path_name(path, regex):
    p = re.compile(regex)
    match = p.search(str(path))
    if match is None:
        return None

    return match.group() # returns the matched string

def load_zStack(zStack):
    ims = imageio.imread(str(zStack))
    imgs0 = []
    for im in ims:
        imgs0.append(im)
    centerIm = imgs0[3]
    # exclude the outer images of the z-stack (do not significantly improve calibration)
    imgs0 = imgs0[1:-1]
    imgs0 = np.array(imgs0)

    return imgs0, centerIm

def write_to_png(array, name):
    imageio.imwrite(array, extension=".png")

class CalibrationData:
    def __init__(self, dataDirectory):
        self.set_data_directory(dataDirectory)
        self.DO_CROP = False
        #self.cropTarget = pathlib.Path(tempfile.TemporaryDirectory().name)
        self.cropTarget = pathlib.Path(self.dataDirectory.name + "_cropped")

    def set_data_directory(self, dir):
        self.dataDirectory = dir
        self._set_actuator_list()

    def _set_actuator_list(self):
        self.actuatorList = [x for x in self.dataDirectory.iterdir() if x.is_dir()]

    def check_images(self):
        for actuator in self.actuatorList:
            calibrationImageSets = list(actuator.glob('*.tif*'))
            for zStack in calibrationImageSets:
                centerIm = imageio.imread(str(zStack))[3]
                if centerIm.shape[0] != centerIm.shape[1]:
                    self._suggest_crop_for_shape()    

                if centerIm.shape[0] > 512:
                    self._suggest_crop_for_size()

        if self.DO_CROP:
            self.dataDirectory = self.set_data_directory(self.cropTarget)

    def _suggest_crop(self, suggestion):
        if not self.DO_CROP:
            prompt = input(suggestion)
            if (prompt == 'n' or prompt == 'N'):
                sys.exit(1)
                print("Exiting.")
            elif (prompt == 'y' or prompt == 'Y' or prompt == ''):
                self.DO_CROP = True
                target = input(f"Specify a path to saved the cropped images "
                               f"(default {self.cropTarget}): ")
                if target == "":
                    target = self.cropTarget.name

                if not os.path.exists(target):
                    os.mkdir(target)

                self.cropTarget = pathlib.Path(target)

                img_crop.crop_images(self.dataDirectory, target, self.actuatorList)
            else:
                print("Invalid response. Exiting.")
                sys.exit(1)


    def _suggest_crop_for_shape(self):
        self._suggest_crop(f"Currently only supports square images. "
                            f"Would you like to automatically crop the images (Y/n)? ")

    def _suggest_crop_for_size(self):
        self._suggest_crop("Warning: using large images will result in significantly "
                            "slower run times. Consider using img-crop.py to crop the "
                            "images.")

    def get_images(self, index):
        return list(self.actuatorList[index].glob('*.tif*'))
