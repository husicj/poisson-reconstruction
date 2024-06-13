import sys
import unittest

import astropy.units as unit
import numpy as np
import scipy.signal as sig

sys.path.append('..')

from aberration import Aberration, ZernikeAberration
from data_loader import MicroscopeParameters
from image import MicroscopeImage


class TestAberrationMethods(unittest.TestCase):
    def setUp(self):
        self.test_microscope0 = MicroscopeParameters(wavelength=0.24 * unit.um)
        self.test_microscope1 = MicroscopeParameters(wavelength=0.4 * unit.um)
        self.test_microscope2 = MicroscopeParameters(wavelength=0.1 * unit.um)
        self.test_aberration0 = Aberration(lambda u, v: 0 * u * v, 4)
        self.test_aberration1 = Aberration(lambda u, v: u * v, 4)
        self.test_aberration2 = Aberration(lambda u, v: 0 * u * v + 1, 4)
        self.test_aberration3 = Aberration(lambda u, v: 0 * u * v, 7)

    def test_gpf(self):
        EXPECTED_RESULT0 = np.fft.ifftshift([[0,0,1,0],
                                             [0,1,1,1],
                                             [1,1,1,1],
                                             [0,1,1,1]])
        OUTPUT0 = self.test_aberration0.gpf(self.test_microscope0)
        self.assertTrue(np.array_equal(OUTPUT0, EXPECTED_RESULT0))

        pix = np.exp(1j * self.test_aberration1._pixel_to_pupil_coordinate(1, self.test_microscope0)**2)

        EXPECTED_RESULT1 = np.fft.ifftshift([[0,        0, 1,        0],
                                             [0, pix ** 1, 1, pix **-1],
                                             [1,        1, 1,        1],
                                             [0, pix **-1, 1, pix ** 1]])
        OUTPUT1 = self.test_aberration1.gpf(self.test_microscope0)
        # some calculation, possibly the np.exp, results in very slightly
        # different values, so np.isclose is used
        self.assertTrue(np.all(np.isclose(OUTPUT1, EXPECTED_RESULT1)))

        EXPECTED_RESULT2 = np.fft.ifftshift([[         0,          0, np.exp(1j),          0],
                                             [         0, np.exp(1j), np.exp(1j), np.exp(1j)],
                                             [np.exp(1j), np.exp(1j), np.exp(1j), np.exp(1j)],
                                             [         0, np.exp(1j), np.exp(1j), np.exp(1j)]])
        OUTPUT2 = self.test_aberration2.gpf(self.test_microscope0)
        self.assertTrue(np.all(np.isclose(OUTPUT2, EXPECTED_RESULT2)))

        OUTPUT3 = self.test_aberration3.gpf(self.test_microscope1)
        EXPECTED_RESULT3 = np.fft.ifftshift([[0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 1, 1, 1, 0, 0],
                                             [0, 0, 1, 1, 1, 0, 0],
                                             [0, 0, 1, 1, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(OUTPUT3, EXPECTED_RESULT3))

        EXPECTED_RESULT4 = np.fft.ifftshift([[1,1,1,1],
                                             [1,1,1,1],
                                             [1,1,1,1],
                                             [1,1,1,1]])
        OUTPUT4 = self.test_aberration0.gpf(self.test_microscope2)
        self.assertTrue(np.array_equal(OUTPUT4, EXPECTED_RESULT4))

    def test_psf(self):
        EXPECTED_RESULT0 = (1/16 * np.array([[11, 3,-1, 3],
                                             [ 3,-1,-1,-1],
                                             [-1,-1, 3,-1],
                                             [ 3,-1,-1,-1]])) ** 2
        OUTPUT0 = self.test_aberration0.psf(self.test_microscope0)
        self.assertTrue(np.array_equal(OUTPUT0, EXPECTED_RESULT0))

        EXPECTED_RESULT2 = (1/16 * np.array([[11, 3,-1, 3],
                                             [ 3,-1,-1,-1],
                                             [-1,-1, 3,-1],
                                             [ 3,-1,-1,-1]])) ** 2
        OUTPUT2 = self.test_aberration2.psf(self.test_microscope0)
        self.assertTrue(np.all(np.isclose(OUTPUT2, EXPECTED_RESULT2)))

        EXPECTED_RESULT3 = np.array([[1,0,0,0],
                                     [0,0,0,0],
                                     [0,0,0,0],
                                     [0,0,0,0]])
        OUTPUT3 = self.test_aberration0.psf(self.test_microscope2)
        self.assertTrue(np.array_equal(OUTPUT3, EXPECTED_RESULT3))

    def test___mul__(self):
        # uses gpf calcultion to determine equivalence, so can fail if
        # test_gpf fails
        mul_aberration0 = self.test_aberration0 * self.test_aberration1
        pix = np.exp(1j * self.test_aberration1._pixel_to_pupil_coordinate(1, self.test_microscope0)**2)
        EXPECTED_RESULT0 = np.fft.ifftshift([[0,        0, 1,        0],
                                             [0, pix ** 1, 1, pix **-1],
                                             [1,        1, 1,        1],
                                             [0, pix **-1, 1, pix ** 1]])
        OUTPUT0 = mul_aberration0.gpf(self.test_microscope0)
        self.assertTrue(np.all(np.isclose(OUTPUT0, EXPECTED_RESULT0)))

        mul_aberration1 = 2 * self.test_aberration0
        EXPECTED_RESULT1 = np.fft.ifftshift([[0,0,1,0],
                                             [0,1,1,1],
                                             [1,1,1,1],
                                             [0,1,1,1]])
        OUTPUT1 = mul_aberration1.gpf(self.test_microscope0)
        self.assertTrue(np.array_equal(OUTPUT1, EXPECTED_RESULT1))

    def test_apply(self):
        np.random.seed(1)
        TEST_IMAGE0 = MicroscopeImage(np.random.rand(4,4),
                                     microscope_parameters=self.test_microscope2)
        OUTPUT0 = self.test_aberration0.apply(TEST_IMAGE0, True)
        PSF = self.test_aberration0.psf(self.test_microscope2)
        self.assertTrue(np.all(np.isclose(OUTPUT0, TEST_IMAGE0)))

        TEST_IMAGE1 = MicroscopeImage(np.random.rand(4,4),
                                     microscope_parameters=self.test_microscope0)
        PSF = self.test_aberration0.psf(self.test_microscope0)
        EXPECTED_RESULT1 = sig.convolve2d(TEST_IMAGE1, PSF, boundary='wrap')[0:4, 0:4]
        OUTPUT1 = self.test_aberration0.apply(TEST_IMAGE1, True)
        self.assertTrue(np.all(np.isclose(EXPECTED_RESULT1, OUTPUT1)))


class TestZernikeAberrationMethods(unittest.TestCase):

    def test_coefficients_to_function(self):
        pass

    def test___mul__(self):
        pass

if __name__ == '__main__':
    unittest.main()
