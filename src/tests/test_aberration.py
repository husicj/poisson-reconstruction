import sys
import unittest

import astropy.units as u
import numpy as np

sys.path.append('..')

from aberration import Aberration, ZernikeAberration
from data_loader import MicroscopeParameters


class TestAberrationMethods(unittest.TestCase):
    def setUp(self):
        self.test_microscope0 = MicroscopeParameters(wavelength=0.24 * u.um)
        self.test_aberration0 = Aberration(lambda x, y: 0 * x * y, 4)
        self.test_aberration1 = Aberration(lambda x, y: x * y, 4)
        self.test_aberration2 = Aberration(lambda x, y: 0 * x * y + 1, 4)

    def test_gpf(self):
        EXPECTED_RESULT0 = [[0,0,1,0],
                            [0,1,1,1],
                            [1,1,1,1],
                            [0,1,1,1]]
        OUTPUT0 = self.test_aberration0.gpf(self.test_microscope0)
        self.assertTrue(np.array_equal(OUTPUT0, EXPECTED_RESULT0))

        pix = self.test_aberration1._pixel_to_pupil_coordinate(1, self.test_microscope0)
        ab_per_pix = pix * np.exp(np.pi * 1j)
        EXPECTED_RESULT1 = [[ 4 * ab_per_pix,  2 * ab_per_pix,  0 * ab_per_pix, -2 * ab_per_pix],
                            [ 2 * ab_per_pix,  1 * ab_per_pix,  0 * ab_per_pix, -1 * ab_per_pix],
                            [ 0 * ab_per_pix,  0 * ab_per_pix,  0 * ab_per_pix,  0 * ab_per_pix],
                            [-2 * ab_per_pix, -1 * ab_per_pix,  0 * ab_per_pix,  1 * ab_per_pix]]
        OUTPUT1 = self.test_aberration1.gpf(self.test_microscope0)
        OUTPUT1.show()
        # self.assertTrue(np.array_equal(OUTPUT1, EXPECTED_RESULT1))

        OUTPUT2 = self.test_aberration2.gpf(self.test_microscope0)

    def test_psf(self):
        EXPECTED_RESULT0 = np.zeros((4,4))
        OUTPUT0 = self.test_aberration0.psf(self.test_microscope0)
        self.assertTrue(np.array_equal(OUTPUT0, EXPECTED_RESULT0))

        OUTPUT1 = self.test_aberration1.psf(self.test_microscope0)

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
