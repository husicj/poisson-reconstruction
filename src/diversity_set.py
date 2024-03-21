import imageio

import data_loader

from aberration import Aberration
from data_loader import MicroscopeParameters
from image import MicroscopeImage
from typing import List

class DiversitySet:
    """This class contains a set of images, represented as a list of instances
    of the image.MicroscopeImage class, contained in the images attribute of
    this class. Each of these MicroscopeImage instances, in addition to
    representing an image, contains an attribute microscope_parameters,
    an instance of the data_loader.MicroscopeParameters class. is consistent
    between all of the images, and match the microscope_parameters attribute of
    this class.

    In addition to the __init__() constructor, this class provides the class
    methods DiversitySet.load() and DiversitySet.load_with_data_loader() for
    constructing an instance from a variety of types of image files, providing
    the microscope parameters; and using data_loader.ExperimentalDataset()
    constructor respectively."""

    def __init__(self,
                 images: np.ndarray,
                 aberrations: List[Aberration],
                 microscope_parameters: None | MicroscopeParameters = None,
                 center_index: int = 0):
        if len(aberrations) != images.shape[2]:
            print("Warning: number of aberrations provided not equal to number"
                  " of images.")
        self.microscope_parameters = microscope_parameters
        self.images = []
        for i in range(images.shape[2]):
            image = MicroscopeImage(images[:,:,i], microscope_parameters, aberrations[i])
            self.images.append(image)
        self.image_count = len(self.images)
        self.center_index = center_index

    @classmethod
    def load(cls,
             path,
             aberrations: np.ndarray | List[Aberration],
             microscope_parameters: None | MicroscopeParameters = None,
             center_index: int = 0):
        """Loads a multi-channel image where each channel corresponds to one of the provided aberrations."""
        images = imageio.imread(path)
        if images.ndim == 2:
            raise TypeError(f"The image at {path} has only a single channel.")
        return cls(images, aberrations, center_index)

    @classmethod
    def load_with_data_loader(cls, data_dir, iteration_number=1):
        """Loads data from a specific existing file structure scheme."""
        data = data_loader.ExperimentalDataset(data_dir, iteration_number)
        images = np.insert(data.phase_diversity_images, 0, data.aberrated_image, axis=0)
        aberration_list = np.insert(data.phase_diversities_coeffs, 0, None, axis = 0)
        aberrations = ZernikeAberration.aberration_list(aberration_list)
        return cls(images, aberrations, data.microscope_parameters, 0)

    def aberrations(self):
        """Returns the applied aberrations of the diverisity images as a list."""
        aberration_list = []
        for image in self.images:
            aberration_list.append(image.aberration)
        return aberration_list

    def ffts(self):
        """Returns the Fourier transforms of each of the diversity images."""
        pass

    def show(self):
        """Display a preview of the image stack."""
        fig = plt.figure(figsize=(32, 6))
        gs = gridspec.GridSpec(1, self.image_count)
        # Plot each image in stack
        for i in range(self.image_count):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(self[:,:,i], cmap='gray')
            ax.set_title(f"Image {i}")
            ax.axis('off')
            if colorbar:
                ax.colorbar()
        plt.show()

    def save(self):
        pass

    ### Basic operations are defined here as component-wise operations ###
    ### on the elements of self.images.                                ###
    def __add__(self, other) -> DiversitySet:
        ret = self
        for i in range(self.image_count):
            ret.images[i] = self.images[i] + other
        return ret

    def __radd__(self, other) -> DiversitySet:
        ret = self
        for i in range(self.image_count):
            ret.images[i] = other + self.images[i]
        return ret

    def __sub__(self, other) -> DiversitySet:
        ret = self
        for i in range(self.image_count):
            ret.images[i] = self.images[i] - other
        return ret

    def __rsub__(self, other) -> DiversitySet:
        ret = self
        for i in range(self.image_count):
            ret.images[i] = other - self.images[i]
        return ret

    def __mul__(self, other) -> DiversitySet:
        ret = self
        for i in range(self.image_count):
            ret.images[i] = self.images[i] * other
        return ret

    def __rmul__(self, other) -> DiversitySet:
        ret = self
        for i in range(self.image_count):
            ret.images[i] = other * self.images[i]
        return ret

    def __div__(self, other) -> DiversitySet:
        ret = self
        for i in range(self.image_count):
            ret.images[i] = self.images[i] / other
        return ret

    def __rdiv__(self, other) -> DiversitySet:
        ret = self
        for i in range(self.image_count):
            ret.images[i] = other / self.images[i]
        return ret
