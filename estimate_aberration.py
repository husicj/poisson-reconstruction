try:
    import imageio.v3 as imageio
except (ImportError, ModuleNotFoundError):
    import imageio
    
import csv
import functools
import glob
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import re
import scipy
import sys
import tifffile

from datetime import datetime
from src import data_loader
from src import fast_fft
from src import file_io
from src import image_functions
from src import iterate_poisson
from src import zern as zern_mod

#####
##### DEFINE CONSTANTS
#####

from parameters import NUM_C, PUPIL_SIZE, REF_INDEX, PIXEL_SIZE, L, NA, NUM_SKIP
LOG_RESULTS = False
CSV_LOG = False
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

def reorder_coefficients(coefficients):
    # reorders coefficients from ordering given by ANSI indexing to Noll indexing
    out = np.zeros(coefficients.shape)
    for noll_j in range(len(out)):
        ansi_j = zern_mod.zernnoll2ansi(noll_j, offset=NUM_SKIP)
        out[noll_j] = coefficients[ansi_j]
    return out

def apply_algorithm(data, logFile=None, verbose=False):
    zStack = np.insert(data.phase_diversity_images, 0, data.aberrated_image, axis=0)
    imgs0, centerIm = zStack, zStack[0]

    aberration_coefficients = np.insert(data.phase_diversities_coeffs, 0, np.zeros(NUM_C), axis = 0)
    print(f"ANSI order:\n{aberration_coefficients}")
    for i in range(len(aberration_coefficients)):
        aberration_coefficients[i] = reorder_coefficients(aberration_coefficients[i])
    print(f"Noll order:\n{aberration_coefficients}")

    dsize = centerIm.shape[0]
    zern = zern_mod.Zernikes(dsize, PUPIL_SIZE, PIXEL_SIZE, num_phi, numskip=NUM_SKIP)
    num_inds = len(zern.inds[0])
    ff = cache_fft(dsize, 1 + data.get_number_of_phase_diversities())

    theta = image_functions.get_theta(aberration_coefficients, zern.zern)

    # imgs = sim_im_2(ob, dim, phi, num_imgs, theta, zern, R, inds, pad=False)
    c0 = np.zeros((NUM_C))+1e-10

    c2, cost2, num_iter2, sss = iterate_poisson.iter_p(zern, imgs0, theta, Sk0, c0.copy(), ff, show=verbose)
    estimate = sss[1]
    plt.imshow(centerIm)
    plt.show()
    plt.imshow(estimate)
    plt.show()

    # log(logFile, f"c[{v}] = {list(np.array(c2))}")
    # c[v] = np.array(c2)

    if not os.path.exists("estimates"):
        os.mkdir("estimates")
    # estimateFile = zStack.parts[-2] + "_" + zStack.parts[-1]
    # with open(f"estimates/{estimateFile}", "wb") as f:
    #     tifffile.imwrite(f, estimate)
    #     log(logFile, f"\n# Image estimate written to {f}\n")

    return c2
        
def square_ob(dsize, squaresize):
    array = np.zeros((2*dsize, 2*dsize))
    lowerbound = dsize - squaresize // 2
    upperbound = lowerbound + squaresize
    array[lowerbound : upperbound , lowerbound : upperbound] = np.ones((squaresize, squaresize))
    print(f"{array.sum()=}")
    return array

        
#####
##### SETUP
#####

data_dir = 'data_dir/'
# iteration_number = 1
data = data_loader.ExperimentalDataset(data_dir, iteration_number=1)

# dsize = 256
# ob = square_ob(dsize, dsize//2)
# TODO structure the appropriate arguments for the following function
# data = data_loader.MockData(imgs, aberration_coeffs, diversity_coeffs)

# plt.imshow(ob)
# plt.show()

if CSV_LOG:
    csvFilePath = "estimated_coefficients.csv"
    if os.path.exists(csvFilePath):
        raise FileExistsError(f"A file already exists at {csvFilePath}.")

if LOG_RESULTS:
    logFilePath = 'estimated_coefficients.py'
    if os.path.exists(logFilePath):
        raise FileExistsError(f"A file already exists at {logFilePath}.")
    print(f"Logging results at {logFilePath}...")
else:
    logFilePath = os.devnull

if (not CSV_LOG) and (not LOG_RESULTS):
    print("\nWarning: Not logging results. Change LOG_RESULTS and/or CSV_LOG to True to do so.\n")

if not os.path.exists("estimates"):
    os.mkdir("estimates")

_ = jnp.array([]) # simplest way I could find to initialize jax

#####
##### RUN ALGORITHM
#####

with open(logFilePath, "a") as logFile:
    log(logFile, f"\n# {datetime.now()}")
    log(logFile, f"# Estimated Zernike coefficients for image at {data_dir}\n")
    log(logFile, f"coefficientList = {{}}\nc = {{}}\n")


    results = apply_algorithm(data, logFile=logFile, verbose=VERBOSE)

    if CSV_LOG:
        with open(csvFilePath, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
