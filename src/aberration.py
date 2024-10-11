from __future__ import annotations

import sys

import numpy as np
import functools
import scipy
import typing

from astropy import units

from data_loader import MicroscopeParameters
from image import DataImage, MicroscopeImage
from fast_fft import Fast_FFTs

class Aberration:
    """Stores an image aberration dictated by the aberration_function and size
    provided to the constructor. This is primarily intended to be used with the
    apply() method, which applies the aberration to a given image. Note that
    the aberration function is expected to be defined on the unit disc as
    opposed to a disc of radius NA/wavelength; conversion to this domain
    occurs during the calculation of the pupil function."""

    def __init__(self,
                 aberration_function: typing.Callable,
                 size: int,
                 ffts: Fast_FFTs = None,
                 ) -> None:
        self.aberration_function = aberration_function
        self.size = size
        if ffts is None:
            self.ffts = Fast_FFTs(size, 1)
        else:
            self.ffts = ffts
        self.microscope = None
        self.gpf_ = None
        self.psf_ = None

    def apply(self,
              image: MicroscopeImage,
              return_real_space_image: bool = False
              ) -> DataImage:
        """Applies the aberration to image, returning the Fourier transform of
        the result by default. The provided image can either be a real space
        image, or the Fourier transform of one, with image.fourier_space == True."""

        if image.fourier_space:
            print("Warning: Aberration.apply() assumes that when it is passed "
                  "a Fourier space image, that is the transform of the image "
                  "to which the aberration is to be applied to reduce the "
                  "number of transforms that must be calculated.")
            F = image
        else:
            F = image.fft(self.ffts)
        S = self.psf(image.microscope_parameters).fourier_transform
        G = F * S
        if return_real_space_image:
            aberrated_image = MicroscopeImage(np.fft.ifftshift(G.fft(self.ffts).real),
                                              image.ffts,
                                              image.microscope_parameters,
                                              image.aberration)
            aberrated_image.fourier_space = False
            aberrated_image.fourier_transform = G
            return aberrated_image
        return G

    def gpf(self,
            microscope: MicroscopeParameters,
            ) -> MicroscopeImage:
        """Returns the generalized pupil function associated with the
        aberration. The microscope parameters are used to set the pixel
        scale."""

        if self.gpf_ is not None and self.microscope == microscope:
            return self.gpf_
        grid = np.mgrid[0:self.size, 0:self.size] # an array of pixel coordinates 
        disc_grid = self._pixel_to_unit_disc_coordinate(grid, microscope)
        array = np.exp(1j * self.aberration_function(disc_grid[0], disc_grid[1]))
        r_grid = np.sqrt((disc_grid ** 2).sum(axis=0))
        pupil = (r_grid <= 1)
        gpf = MicroscopeImage(np.fft.ifftshift(pupil * array), self.ffts, microscope, None)
        gpf.fourier_space = True
        self.gpf_ = gpf
        return gpf
    
    def get_phase(self,
                    microscope: MicroscopeParameters
                    ) -> MicroscopeImage:
        """Returns the phase of the generalized pupil function associated with the
        aberration. The microscope parameters are used to set the pixel
        scale."""
        grid = np.mgrid[0:self.size, 0:self.size] # an array of pixel coordinates 
        disc_grid = self._pixel_to_unit_disc_coordinate(grid, microscope)
        array = self.aberration_function(disc_grid[0], disc_grid[1])
        r_grid = np.sqrt((disc_grid ** 2).sum(axis=0))
        pupil = (r_grid <= 1)
        phase = MicroscopeImage(np.fft.ifftshift(pupil * array), self.ffts, microscope, None)
        phase.fourier_space = True
        return phase



    def _pixel_to_pupil_coordinate(self,
                                   pixel_indices: np.ndarray,
                                   microscope: MicroscopeParameters
                                   ) -> np.ndarray:
        """Converts a pixel index value to a pupil plane coordinate,
        based on the relevant microscope parameters."""

        scale_factor = 0.5 / (microscope.pixel_size.value * (self.size // 2))
        return scale_factor * (pixel_indices - self.size // 2)

    def _pupil_to_unit_disc_coordinate(self,
                                       pupil_coordinates: np.ndarray,
                                       microscope: MicroscopeParameters
                                       ) -> np.ndarray:
        """Converts a pupil plane coordinate to a corresponding unit disc
        coordinate: the support of the pupil function is scaled to the unit
        disc."""

        NA = microscope.numerical_aperture
        wavelength = microscope.wavelength.value
        return pupil_coordinates * wavelength / NA

    def _pixel_to_unit_disc_coordinate(self,
                                       pixel_indices: np.ndarray,
                                       microscope: MicroscopeParameters
                                       ) -> np.ndarray:
        pupil_coordinates = self._pixel_to_pupil_coordinate(pixel_indices,
                                                            microscope)
        return self._pupil_to_unit_disc_coordinate(pupil_coordinates,
                                                   microscope)

    def psf(self,
            microscope: MicroscopeParameters
            ) -> MicroscopeImage:
        """Returns the point spread function s associated with the given
        generalized pupil function. The Fourier transform of the psf is also
        computed, and returned as an attribute (fourier_transform) of the
        returned psf."""

        if self.psf_ is not None and self.microscope == microscope:
            return self.psf_
        h = self.gpf(microscope).fft()
        s = np.fft.fftshift((np.abs(h)**2).real).view(DataImage)
        s.fourier_space = False
        S = s.fft(self.ffts)
        # TODO these might be swapped
        S.fourier_space = True
        s.fourier_transform = S
        self.psf_ = s
        self.microscope = microscope
        return s

    def __mul__(self, other):
        if other is None:
            return self
        if not isinstance(other, Aberration):
            def combined_aberration_function(x, y):
                return other * self.aberration_function(x,y)
        else:
            def combined_aberration_function(x, y):
                return (self.aberration_function(x,y) +
                        other.aberration_function(x,y))
        out = Aberration(combined_aberration_function,
                          self.size,
                          self.ffts)
        return out

    def __rmul__(self, other):
        if other is None:
            return self
        return self.__mul__(other)

    def __sizeof__(self):
        size = 0
        for attribute in dir(self):
            if isinstance(attribute, np.ndarray):
                size += sys.getsizeof(attribute.base)
            else:
                size += sys.getsizeof(attribute)
        return size

class ZernikeAberration(Aberration):
    """Stores an image aberration specified by coefficients of Zernike
    polynomials and the size provided to the constructor.

    It is important to note that although the 0th and 1st order Zernike
    polynomials represent aberrations of phase only, but not intensity,
    and so have no impact on an image taken in incoherent light source methods.
    However, the methods of this class assume the presence of these
    coefficients regardless, for consistency. Therefore, coefficient sets that
    do not include these, typically the first three coefficients in a single
    index scheme, should be padded with leading 0s before being passed to any
    of the methods of this class."""

    def __init__(self,
                 coefficients: np.ndarray | List,
                 size: int,
                 ffts: Fast_FFTs = None,
                 indexing: str = "Noll"
                 ) -> None:
        if indexing == "Noll":
            self.coefficients = np.array(coefficients)
        else:
            self.coefficients = self.coefficients_to_noll(coefficients, indexing)
        aberration_function = self.coefficients_to_function(self.coefficients)
        super().__init__(aberration_function, size, ffts)

    @classmethod
    def aberration_list(cls,
                        aberrations: np.ndarray | List[np.ndarray],
                        size: int,
                        ffts: Fast_FFTs = None,
                        indexing: str = "Noll"
                        ) -> List[ZernikeAberration]:
        """A class method that takes an array or list of sets of Zernike
        polynomial coefficients and returns a list of objects of the called
        class determined by these coefficients."""

        list_ = []
        for aberration in aberrations:
            list_.append(cls(aberration, size, ffts, indexing))
        return list_

    def coefficients_to_function(self,
                                 coefficients: list | np.ndarray
                                 ) -> typing.Callable:
        """Returns an aberration function determined by the given coefficients
        of Zernike polynomials, using the Noll indexing order."""

        zernike_terms_list = []
        for j, coefficient in enumerate(coefficients):
            n, m = self.zernnoll2nm(j, numskip=0)
            func = functools.partial(self.zernike, n, m)
            zernike_terms_list.append((coefficient, func))
        def aberration_function(x, y):
            acc = 0
            for coefficient, func in zernike_terms_list:
                acc += coefficient * func(x, y)
            return acc
        return aberration_function

    def coefficients_to_noll(self,
                             coefficients: list | np.ndarray,
                             indexing: str
                             ) -> np.ndarray:
        """Converts coefficients from various indexing schemes of the Zernike
        polynomials to the Noll ordering."""

        match indexing:
            case "Noll":
                return coefficients
            case "ANSI":
                return self.ansi_to_noll(coefficients)
            case _:
                raise ValueError("Invalid indexing scheme provided.")

    def ansi_to_noll(self, coefficients: list | np.ndarray) -> np.ndarray:
        """Converts coefficients from ANSI indexing scheme to Noll"""

        def ansi_to_zernike_indices(j):
            # In case that j is a list convert it to numpy array
            j = np.array(j)
            n = (np.ceil((-3 + np.sqrt(9+8*j))/2))
            m = 2*j-n*(n+2)
            return n.astype(int), m.astype(int)
        
        def zernike_indices_to_noll(n, m):
            # Convert the indices to Noll
            j = (n * (n + 1)) // 2 + np.abs(m)
            
            # Corrections
            mask1 = (m >= 0) & (n % 4 >= 2)
            mask2 = (m <= 0) & (n % 4 <= 1)
            
            j[mask1 | mask2] += 1
            return j

        n, m = ansi_to_zernike_indices(np.arange(len(coefficients)))
        noll_indices = zernike_indices_to_noll(n, m)
        return coefficients[noll_indices-1]


    def noll(self):
        pass

    def ansi(self):
        pass

    def __mul__(self, other):
        if isinstance(other, type(self)):
            coefficients = self.coefficients + other.coefficients
            return ZernikeAberration(coefficients, self.size, self.ffts)

    def __add__(self, other):
        if isinstance(other, type(self)):
            coefficients = self.coefficients + other.coefficients
            return ZernikeAberration(coefficients, self.size, self.ffts)

    #####
    # The following functions are used to calculate the value of
    # a given Zerike polynomial at at given coordinate (x , y).
    # The Zernike polynomials are typically defined on the unit disc,
    # and here are constantly zero outside of the unit disc.
    # Note that these conform to the general definition of Zernike
    # polynomials, but the domain needs to be rescaled to decompose an
    # aberration into Zernike components, since the pupil is not typically
    # of radius 1, but has radius given by numerical aperture / wavelength

    def zern_theta(self, *args):
        if len(args) == 2:
            m, theta = args
        elif len(args) == 3:
            n, m, theta = args
        elif len(args) == 4:
            n, m, r, theta = args
        else:
            raise TypeError("zern_theta() takes from 2 to 4 positional arguments but {} were given"
                            .format(len(args) + 1))
        if m < 0:
            return np.sin(-m * theta)
        else:
            return np.cos(m * theta)


    def zern_polar(self, n, m, r, theta):
        return self.zern_r(n, m, r, theta) * self.zern_theta(n, m, r, theta)

    def zern_r(self, n, m, r, theta = None):
        R = np.asarray(r)
        disc = (R<=1)
        acc = np.zeros(R.shape)
        if (n-m) % 2 == 0:
            mn = (n-np.abs(m)) // 2
            for k in range(mn + 1):
                acc += (-1)**k * scipy.special.comb(n-k, k) * scipy.special.comb(n-2*k, mn-k) * r**(n - 2*k)
        return acc * disc

    def zernike(self, n, m, u, v):
        r = np.sqrt(u**2 + v**2)
        theta = np.arctan2(v, u)
        return self.zern_polar(n, m, r, theta)

    def zernike_pixels(self, n, m, x, y):
        u = self._pixel_to_unit_disc_coordinate(x)
        v = self._pixel_to_unit_disc_coordinate(y)
        return self.zernike(n, m, u, v)

    def zernike_pixel_array(self, j0):
        n, m = self.zernnoll2nm(j0)
        grid = np.mgrid[0:self.size, 0:self.size] # an array of coordinates 
        disc_grid = self._pixel_to_unit_disc_coordinate(grid, self.microscope)
        shifted_array = MicroscopeImage(self.zernike(n, m, disc_grid[0], disc_grid[1]),
                                        self.ffts,
                                        self.microscope,
                                        None)
        shifted_array.fourier_space = True
        shifted_array.zero_frequency_centered = True
        array = shifted_array.ifftshift()
        return array

    def zernnoll2nm(self, j0, numskip=0):  #technically noll -1, so starting at j = 0
        j = j0 + numskip + 1
        indices = np.array(np.ceil((1+np.sqrt(1+8*j))/2),dtype=int)-1
        triangular_numbers = np.array(indices*(indices+1)/2).astype(int)
        n = indices -1
        r = j - triangular_numbers
        r +=n
        m = (-1)**j * ((n % 2) + 2 * np.array((r + ((n+1)%2))/2).astype(int))
        return n, m
    #
    #####

    @staticmethod
    def convert_rms_length_to_phase_units(value, wavelength, unit=None):
        if unit is None:
            unit = units.um
        if type(value) is list:
            value = np.array(value)
        wavelength = wavelength.to(unit).value
        conversion_factor = 2*np.pi / wavelength
        return value * conversion_factor
