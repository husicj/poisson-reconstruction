import sys
import unittest

import astropy.units as unit
import numpy as np

sys.path.append('..')

from aberration import Aberration, ZernikeAberration
from data_loader import MicroscopeParameters


class TestAberrationMethods(unittest.TestCase):
    def setUp(self):
        self.test_microscope0 = MicroscopeParameters(wavelength=0.24 * unit.um)
        self.test_microscope1 = MicroscopeParameters(wavelength=0.4 * unit.um)
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

    def test_psf(self):
        EXPECTED_RESULT0 = (1/16 * np.array([[11, 3,-1, 3],
                                             [ 3,-1,-1,-1],
                                             [-1,-1, 3,-1],
                                             [ 3,-1,-1,-1]])) ** 2
        OUTPUT0 = self.test_aberration0.psf(self.test_microscope0)
        self.assertTrue(np.array_equal(OUTPUT0, EXPECTED_RESULT0))

        OUTPUT1 = self.test_aberration1.psf(self.test_microscope0)
        OUTPUT1.show()

        EXPECTED_RESULT2 = [[1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]
        OUTPUT2 = self.test_aberration2.psf(self.test_microscope0)
        self.assertTrue(np.array_equal(OUTPUT2, EXPECTED_RESULT2))


class TestZernikeAberrationMethods(unittest.TestCase):

    pass

if __name__ == '__main__':
    unittest.main()
