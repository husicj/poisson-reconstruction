from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fast_fft import Fast_FFTs
from bin.mem_profile import show_stack

if TYPE_CHECKING:
    from data_loader import MicroscopeParameters
    from aberration import Aberration


def show_colorbar(ax, mappable=None, label=None, orientation="right", size="5%", pad=0.05, fontsize=12):
    """
    Adds a colorbar to the plot associated with the given axes and mappable.

    Parameters:
    - ax: The matplotlib axes to which to add the colorbar.
    - mappable: The mappable object to which the colorbar is associated.
    - orientation: The orientation of the colorbar ("right" or "top" etc.).
    - size: The size of the colorbar.
    - pad: The padding between the colorbar and the axes.
    """
    if mappable is None:
        mappable = ax.images[0]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(orientation, size=size, pad=pad)
    plt.colorbar(mappable, cax=cax)
    if label is not None:
        cax.set_ylabel(label, fontsize=fontsize)


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
        
    def __array_finalize__(self, obj):
        if obj is None: return
        super().__array_finalize__(obj)
        self.ffts = getattr(obj, 'ffts', None)
        self.fourier_space = getattr(obj, 'fourier_space', False)
        self.zero_frequency_centered = getattr(obj, 'zero_frequency_centered', False)

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
              *args, **kwargs):
        """Creates a blank image, filled with ones (using numpy.ones).
        The *args are additional arguments passed to the __new__() method, used
        by subclasses of DataImage."""
        try:
            dtype = kwargs['dtype']
        except KeyError:
            dtype = None
        image = np.ones((size, size), dtype=dtype)
        return cls(image, ffts, *args)

    def crop(self, size: int, ffts: None | Fast_FFTs = None):
        """Returns an object of the same type, but cropped to the given size
        about the center of the image."""

        if size > self.shape[0]:
            raise ValueError("Cropped size larger than original")
        crop_slice = slice(self.shape[0]//2 - size//2, self.shape[0]//2 + size//2)
        # the image is cast to an ndarray and then a new DataImage
        # is created because the FFT matrix is not shared
        # Note: this means that _crop_return_func() should be overwritten for subclasses
        cropped_image_array = self[crop_slice, crop_slice].view(np.ndarray)
        return self._crop_return_func(cropped_image_array, ffts)

    def _crop_return_func(self, cropped_image_array, ffts):
        """A helper method for crop() to handle different arguments for subclassing."""
        return DataImage(cropped_image_array, ffts)

    def fft(self,
            ffts: None | Fast_FFTs = None,
            force_forward: Bool = False,
            force_inverse: Bool = False,
            force_recalculate: Bool = False):
        """Calculates the fourier transform of the image, caching the result.
        By default, the forward or inverse transform is chosen based on the
        value of self.fourier_space. For instances when the transform is not
        between two literal images, this can be overridden by the forcing
        arguments."""
        if getattr(self, 'ffts', None) is None:
            if ffts is not None:
                self.ffts = ffts
            else:
                self.ffts = Fast_FFTs(self.shape[0], 1)
        if force_forward:
            out = self.ffts.fft(self).view(DataImage)
            out.fourier_space = None
            return out
        if force_inverse:
            out = self.ffts.ift(self).view(DataImage)
            out.fourier_space = None
            return out
        if (getattr(self, 'fourier_transform', None) is None) or force_recalculate:
            if self.fourier_space:
                self.fourier_transform = self.ffts.ift(self).view(DataImage)
            else:
                self.fourier_transform = self.ffts.fft(self).view(DataImage)
        self.fourier_transform.fourier_space = not self.fourier_space
        if self.zero_frequency_centered and not self.fourier_space:
            return (self.fourier_transform).fftshift()
        else:
            return self.fourier_transform

    def fftshift(self, axes=None):
        if not self.fourier_space:
            print("Warning: applying fftshift to a non-Fourier space image.")
            show_stack()
        if self.zero_frequency_centered:
            print("Warning: using fftshift where ifftshift is probably "
                  "correct instead")
            show_stack()
        out = np.fft.fftshift(self, axes).view(DataImage)
        out.zero_frequency_centered = not self.zero_frequency_centered
        return out

    def ifftshift(self, axes=None):
        if not self.fourier_space:
            print("Warning: applying fftshift to a non-Fourier space image.")
        if not self.zero_frequency_centered:
            print("Warning: using ifftshift where fftshift is probably "
                  "correct instead")
            show_stack()
        out = np.fft.ifftshift(self, axes).view(DataImage)
        out.zero_frequency_centered = not self.zero_frequency_centered
        return out

    def save(self, path):
        pass

    def show(self):
        """Plots the image represented by the class."""
        if self.fourier_space:
            if self.zero_frequency_centered:
                obj = self
            else:
                print(f"{type(self)=}")
                print(self.zero_frequency_centered)
                obj = self.fftshift()
                print(type(obj))
            print("Fourier space images are shown with the 0 frequency component at the center.")
        else:
            obj = self
        if np.iscomplexobj(self):
            fig, (ax0, ax1) = plt.subplots(1,2)
            ax0.imshow(obj.real, cmap = 'gray')
            ax1.imshow(obj.imag, cmap = 'gray')
            ax0.set_title('real component')
            ax1.set_title('imaginary component')
        else:
            plt.imshow(obj, cmap='gray')
            show_colorbar(plt.gca(), label='Intensity')
        plt.show()

    def __mul__(self, other):
        if isinstance(other, DataImage):
            if self.fourier_space != other.fourier_space:
                if self.fourier_space is not None and other.fourier_space is not None:
                    print("Warning: coordinate space mismatch for multiplication.")
                    show_stack()
            if (self.fourier_space and 
                (self.zero_frequency_centered != other.zero_frequency_centered)):
                print("Warning: computing a pointwise product of fourier space "
                      "images where the zero-frequency components are "
                      "not aligned. This can be resolved by calling the "
                      "fftshift or ifftshift methods or one of the images. "
                      "Note that using numpy.fft.fftshift or np.fft.ifftshiftt "
                      "will not resolve this warning as it does not modify the "
                      "image's zero_frequency_centered attribute.")
                show_stack()
        out = super().__mul__(other)
        _ = out.fft(force_recalculate = True)
        if getattr(other, 'fourier_space', None) is None:
            out.fourier_space = None
        else:
            out.fourier_space = self.fourier_space
        return out

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
                ffts:                  None | Fast_FFTs = None,
                microscope_parameters: None | MicroscopeParameters = None,
                aberration:            None | Aberration = None):
        obj = super().__new__(cls, input_array, ffts)
        if microscope_parameters is None:
            print("Warning: No microscope parameters were provided to the"
                  "constructor of class MicroscopeImage class, so DataImage"
                  "type will be returned instead.")
            return DataImage(obj)
        obj.microscope_parameters = microscope_parameters
        obj.aberration = aberration
        return obj

    def _crop_return_func(self, cropped_image_array, ffts):
        """A helper method for crop() to handle different arguments for subclassing."""

        return MicroscopeImage(cropped_image_array,
                               ffts,
                               self.microscope_parameters,
                               self.aberration)

    def __array_finalize__(self, obj):
        if obj is None: return
        super().__array_finalize__(obj)
        self.microscope_parameters = getattr(obj, 'microscope_parameters', None)
        self.aberration = getattr(obj, 'aberration', None)
