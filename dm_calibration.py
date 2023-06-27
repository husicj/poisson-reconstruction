try:
    import imageio.v3 as imageio
except (ImportError, ModuleNotFoundError):
    import imageio
    
import functools
import glob
import jax.numpy as jnp
import numpy as np
import os
import pathlib
import re
import scipy
import sys
import tempfile

from datetime import datetime
from src import fast_fft
from src import file_io
from src import image_functions
from src import iterate_poisson
from src import zern as zern_mod

#####
##### DEFINE CONSTANTS
#####

from parameters import NUM_C, PUPIL_SIZE, REF_INDEX
PIXEL_SIZE = 0.08
L = 0.532
NA = 1.2
DIV_MAG = 1
NUM_ACTUATORS = 3
NUM_IMGS = 5
LOG_RESULTS = True # if this is set to False, no data is recorded; meant to be used for debugging
VERBOSE = True
num_phi = NUM_C
Sk0 = np.pi*(PIXEL_SIZE*PUPIL_SIZE)**2

#####
##### DEFINE FUNCTIONS
#####

def log(logFile, logString):
    if logFile is not None:
        logFile.write(logString + "\n")

@functools.lru_cache(maxsize=None)
def cache_zernikes(dsize, pupilSize, pixelSize, num):
    return zern_mod.get_zern(dsize, pupilSize, pixelSize, num)

@functools.lru_cache(maxsize=None)
def cache_fft(dsize, numImgs):
    return fast_fft.Fast_FFTs(dsize, numImgs, 1)

def invalid_response():
    prompt = input("Please enter either y or n: ")
    if (prompt == 'n' or prompt == 'N'):
        sys.exit()
    if (prompt == 'y' or prompt == 'Y'):
        sourceDirectory = pathlib.Path("sample_data")
    else:
        sourceDirectory = invalid_response()

    return sourceDirectory

def apply_algorithm_for_calibration(images, index, logFile=None, verbose=False):
    for zStack in images:
        v = file_io.scan_path_name(zStack, r'[+-]?\d*([.,]\d+)|[+-]\d+')
        if v is None:
            print(f"Could not determine the calibration voltage for file {zStack}."
                  "Please add the voltage to the filename. Skipping.")
            continue
        
        if verbose:
            print(f"Voltage: {str(v)} (from file '{str(zStack)}')")

        imgs0, centerIm = file_io.load_zStack(zStack)

        defocus_steps = np.array([-2,-1,0,1,2])*DIV_MAG
        dsize = centerIm.shape[0]
        zern = zern_mod.Zernikes(dsize, PUPIL_SIZE, PIXEL_SIZE, num_phi)
        num_inds = len(zern.inds[0])
        ff = cache_fft(dsize, NUM_IMGS)

        theta = image_functions.defocus(defocus_steps, zern.R, zern.inds, NA, L, REF_INDEX)
        F = image_functions.fft2(np.ascontiguousarray(centerIm))
        f = np.ones((dsize, dsize))
        c0 = np.zeros((NUM_C))+1e-10

        c2, cost2, num_iter2, sss = iterate_poisson.iter_p(zern, imgs0, theta, Sk0, c0.copy(), ff, show=verbose)

        log(logFile, f"c[{v}] = {list(np.array(c2))}")

    log(logFile, f"coefficientList[\"{actuator.parts[-1]}\"] = c.copy()\n")

#####
##### SETUP
#####

try:
    sourceDirectory = pathlib.Path(sys.argv[1])
except IndexError:
    prompt = input("No data directory given. To apply this script to data in a local directory, provide the path to the data as an argument when running the program. Alterinatively, proceed with sample data (Y/n)? ")
    if (prompt == 'n' or prompt == 'N'):
        sys.exit()
    if (prompt == 'y' or prompt == 'Y' or prompt == ''):
        sourceDirectory = pathlib.Path("sample_data")
    else:
        sourceDirectory = invalid_response()

_ = jnp.zeros((1)) # simplest way I could find to initialize jax
data = file_io.CalibrationData(sourceDirectory)
data.check_images()

if LOG_RESULTS:
    logFilePath = 'calibration_data_coefficients.py'
    if os.path.exists(logFilePath):
        raise FileExistsError(f"A file already exists at {logFilePath}.")
    print(f"Logging results at {logFilePath}...")
else:
    logFilePath = os.devnull
    print("\nWarning: Not logging results. Change LOG_RESULTS to True to do so.\n")

#####
##### RUN ALGORITHM
#####

with open(logFilePath, "a") as logFile:
    log(logFile, f"\n# {datetime.now()}")
    log(logFile, "# Zernike coefficients of mirror calibration data from src.iterate_poisson.iter_p\n")
    log(logFile, f"coefficientList = {{}}\nc = {{}}\n")
       
    for i in range(len(data.actuatorList)):
        actuator = data.actuatorList[i]
        images = data.get_images(i)
        if not images:
            continue

        print(f"\nActuator: {actuator}")
        actuatorNumber = file_io.scan_path_name(actuator, r'\d+') # tries to extract a number from the directoryName
        if actuatorNumber is None:
            actuatorNumber = 0
            print(f"Warning: Could not guess actuator number from directory name. Using value {actuatorNumber}")

        print(f"Applying Poisson noise model algorithm for actuator {actuator}...")
        apply_algorithm_for_calibration(images, i, logFile=logFile, verbose=VERBOSE)
