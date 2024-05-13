from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from fast_fft import Fast_FFTs
from bin.mem_profile import show_stack

if TYPE_CHECKING:
    from data_loader import MicroscopeParameters
    from aberration import Aberration

class DataImage(np.ndarray):
    """Represents an image, an inherits all methods from the np.ndarray class.
    The input_array must be an object that can be cast to an np.ndarray,
    and the resulting array must have ndim == 2 or ndim == 3, or a TypeError
    is raised."""

    def __new__(cls, input_array, ffts: None | Fast_FFTs = None):
        # Cast input_array as a DataImage type
        obj = np.asarray(input_array).view(cls)
        # Check that input_array has a reasonable shape
        if not (obj.ndim == 2 or obj.ndim == 3):
            raise TypeError(f"Array with dimension {obj.ndim} cannot be cast "
                            f"to type {cls.__name__}. Suitable arrays should "
                            f"cast to an numpy.ndarray with ndim=2 or ndim=3.")
        # Attribute defaults should be assigned in __array_finalize__()
        # see https://numpy.org/doc/stable/user/basics.subclassing.html
        obj.ffts = ffts
        return obj
        
    @classmethod
    def load(cls, path, *args):
        """Loads an image from a file. The *args are additional arguments
        passed to the __new__() method, used by subclasses of DataImage."""
        image = imageio.imread(path)
        return cls(image, *args)

    @classmethod
    def blank(cls,
              size: int, 
              ffts: None | Fast_FFTs = None,
              *args):
        """Creates a blank image, filled with ones (using numpy.ones).
        The *args are additional arguments passed to the __new__() method, used
        by subclasses of DataImage."""
        image = np.ones((size, size))
        return cls(image, ffts, *args)

    def fft(self,
            ffts: Fast_FFTs = None):
        if not hasattr(self, 'ffts'):
            if ffts is not None:
                self.ffts = ffts
            else:
                self.ffts = Fast_FFTs(self.shape[0], 1)
        if not hasattr(self, 'fourier_transform'):
            if self.fourier_space:
                self.fourier_transform = self.ffts.ift(self)
            else:
                self.fourier_transform = self.ffts.fft(self)
        return self.fourier_transform.view(DataImage)

    def save(self, path):
        pass

    def show(self, colorbar=False):
        """Plots the image represented by the class."""
        plt.imshow(self[:,:,i], cmap='gray')
        ax.axis('off')
        if colorbar:
            ax.colorbar()
        plt.show()

    def __array_finalize__(self, obj):
        if obj is None: return
        super().__array_finalize__(obj)
        self.ffts = getattr(obj, 'ffts', None)
        self.fourier_space = getattr(obj, 'fourier_space', False)

    def __sizeof__(self):
        size = self.nbytes
        for attribute in dir(self):
            if isinstance(attribute, np.ndarray):
                size += sys.getsizeof(attribute.base)
            else:
                size += sys.getsizeof(attribute)
        return size

class MicroscopeImage(DataImage):
    """Represents an image with associated microscope data and applied
    aberration, which can be None. The input array can be any type that can be
    cast to an np.ndarray, but the resulting array must have ndim == 2 or
    ndim == 3, or a TypeError will be raised. if None is passed as the
    microscope_parameters argument to the constructor, a DataImage object
    will be returned instead.

    The load() method is inherited from the DataImage class, but must receive
    the microscope_parameters and aberration arguments of the method
    Microscope.__new__() for its *args argument.
    """

    def __new__(cls, input_array,
                microscope_parameters: None | MicroscopeParameters = None,
                aberration:            None | Aberration = None,
                ffts:                  None | Fast_FFTs = None):
        obj = super().__new__(cls, input_array, ffts)
        if microscope_parameters is None:
            print("Warning: No microscope parameters were provided to the"
                  "constructor of class MicroscopeImage class, so DataImage"
                  "type will be returned instead.")
            return DataImage(obj)
        obj.microscope_parameters = microscope_parameters
        obj.aberration = aberration
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        super().__array_finalize__(obj)
        self.microscope_parameters = getattr(obj, 'microscope_parameters', None)
        self.aberration = getattr(obj, 'aberration', None)
