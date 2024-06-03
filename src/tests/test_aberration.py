import unittest

import numpy as np

from aberration import Aberration, ZernikeAberration
from data_loader import MicroscopeParameters

class TestAberrationMethods(unittest.TestCase):
    def setUp(self):
        self.test_microscope0 = MicroscopeParameters()
        self.test_aberration0 = Aberration(lambda x, y: 0 * x * y, 4)
        self.test_aberration1 = Aberration(lambda x, y: x * y, 4)

    def test__pixel_to_pupil_coordinate(self):
        INPUT = np.mgrid[0:4, 0:4]
        EXPECTED_RESULT = 0.532 / (1.2 * 4 * 0.104) * INPUT
        OUTPUT = self.test_aberration0._pixel_to_pupil_coordinate(INPUT, self.test_microscope0)
        self.assertTrue(
                np.array_equal(OUTPUT, EXPECTED_RESULT)
                )

    def test_gpf(self):
        EXPECTED_RESULT0 = np.zeros((4,4))
        OUTPUT0 = self.test_aberration0.gpf(self.test_microscope0)
        self.assertTrue(
                np.array_equal(OUTPUT0, EXPECTED_RESULT0)
                )

        pix = self.test_aberration1._pixel_to_pupil_coordinate(1, self.test_microscope0)
        ab_per_pix = pix * np.exp(np.pi * 1j)
        EXPECTED_RESULT1 = [[ 4 * ab_per_pix,  2 * ab_per_pix,  0 * ab_per_pix, -2 * ab_per_pix],
                            [ 2 * ab_per_pix,  1 * ab_per_pix,  0 * ab_per_pix, -1 * ab_per_pix],
                            [ 0 * ab_per_pix,  0 * ab_per_pix,  0 * ab_per_pix,  0 * ab_per_pix],
                            [-2 * ab_per_pix, -1 * ab_per_pix,  0 * ab_per_pix,  1 * ab_per_pix]]
        OUTPUT1 = self.test_aberration1.gpf(self.test_microscope0)
        self.assertTrue(
                np.array_equal(OUTPUT0, EXPECTED_RESULT0)
                )


class TestZernikeAberrationMethods(unittest.TestCase):

    pass

if __name__ == '__main__':
    unittest.main()
