import numpy as np

# various parameters used across the scripts present in this directory

#
### PARAMETER FUNCTIONS
#

def get_coefficient_count(degree, piston, tip_tilt):
    if degree == 0:
        return int(piston)
    elif degree == 1:
        return get_coefficient_count(0, piston, tip_tilt) + (2 * int(tip_tilt))
    else:
        return degree + 1 + get_coefficient_count(degree - 1, piston, tip_tilt)


#
### SYSTEM PARAMETERS
#

# lengths in units of um
PIXEL_SIZE = 0.08 # size of pixel in image space, its inverse is the size of pixels in frequency space
NA = 1.2 # Numerical aperature
L = 0.532 # wavelength
REF_INDEX = 1.333 # refractive index, using water for now

PUPIL_SIZE = NA/L
#
### RECONSTRUCTION PARAMETERS
#

DIV_MAG = 3*L # diversity magnitude - defocus distance
MAX_ZERNIKE_DEGREE = 5 # Degree of highest order Zernike polynomials used
OMIT_PISTON = True # Omit 0th order Zernike polynomial - almost always going to be True
OMIT_TIP_AND_TILT = False # Omit 1st order Zernike polynomials

# Number of Zernike polynomials used
NUM_C = get_coefficient_count(MAX_ZERNIKE_DEGREE, not OMIT_PISTON, not OMIT_TIP_AND_TILT) 

#
### PLOTTING PARAMETERS
#

VOLTAGES = [-.03, -.015, 0, .015, .03]
ACTUATOR_COUNT = 52
ACTUATOR_ARRANGEMENT = np.array([
    [ 0,  0,  1,  2,  3,  4,  0,  0],
    [ 0,  5,  6,  7,  8,  9, 10,  0],
    [11, 12, 13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24, 25, 26],
    [27, 28, 29, 30, 31, 32, 33, 34],
    [35, 36, 37, 38, 39, 40, 41, 42],
    [ 0, 43, 44, 45, 46, 47, 48,  0],
    [ 0,  0, 49, 50, 51, 52,  0,  0]
]).transpose()
MIRROR_SIZE = 8 # number of actuators at widest point in mirror
PLOT_SIZE = 128
