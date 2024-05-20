import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy import units as u

class MicroscopeParameters:
    def __init__(self, numerical_aperture=1.2, wavelength=0.532*u.um, pixel_size=0.104*u.um):
        self.numerical_aperture = numerical_aperture
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.frequency_scale_factor = pixel_size * wavelength / numerical_aperture
        # self.ID = hash((numerical_aperture, wavelength, pixel_size))

    def __eq__(self, other):
        if other is None:
            return False
        return (self.numerical_aperture == other.numerical_aperture and
                self.wavelength == other.wavelength and
                self.pixel_size == other.pixel_size
                )


class ExperimentalDataset:
    # Load experimental phase diversity data set
    def __init__(self, data_dir, iteration_number=1, microscope_parameters=MicroscopeParameters(1.2, 0.532*u.um, 0.104*u.um)):
        self.data_dir = data_dir
        self.iteration_number = iteration_number
        self.microscope_parameters = microscope_parameters
        gt_image, gt_phase_aberration_coeffs, aberrated_image, phase_diversities_coeffs, phase_diversity_images = self.load_data()
        self.gt_image = gt_image
        self.gt_phase_aberration_coeffs = gt_phase_aberration_coeffs
        self.aberrated_image = aberrated_image
        self.phase_diversities_coeffs = phase_diversities_coeffs
        self.phase_diversity_images = phase_diversity_images

    def get_number_of_phase_diversities(self):
        return self.phase_diversity_images.shape[0]
    
    def get_acquisition_data(self):
        # Aberrated image and phase diversity images
        acquisition_data = np.concatenate((self.aberrated_image[np.newaxis,:,:], self.phase_diversity_images), axis=0)
        return acquisition_data
    
    def get_acquisition_image_width(self):
        return self.aberrated_image.shape[1]

    def load_data(self):

        gt_phase_aberration_coeffs, phase_diversities_coeffs = self.load_phase_coefficients()
        gt_image, aberrated_image, phase_diversity_images = self.load_images()

        return gt_image, gt_phase_aberration_coeffs, aberrated_image, phase_diversities_coeffs, phase_diversity_images
    
    def load_phase_coefficients(self):
        # Load file CoeffsIn.txt that contains the phase aberration coefficients and phase diversity coefficients
        phase_coeffs = np.loadtxt(os.path.join(self.data_dir, 'CoeffsIn.txt'))
        # Discard tip and tilt coefficients (first two columns) as they are supposed to be 0
        phase_coeffs = phase_coeffs[:,2:]

        # GT phase aberration Zernike coefficients are stored in first line of CoeffsIn.txt
        gt_phase_aberration_coeffs = phase_coeffs[0,:]
        # Phase diversity coefficients are stored in line 2 to end of CoeffsIn.txt
        phase_diversities_coeffs = phase_coeffs[1:,:]

        return gt_phase_aberration_coeffs, phase_diversities_coeffs
    
    def load_images(self):
        # Unaberrated image (GT.tif)
        #gt_image = imageio.imread(os.path.join(self.data_dir, 'GT.tif'))
        gt_image = imageio.imread(os.path.join(self.data_dir, f'GT.tif'))
        # Aberrated images (stored in Iteration #/Image_phasediversity.tif)
        images = imageio.imread(os.path.join(self.data_dir, f'Iteration {self.iteration_number}/Stack.tif'))
        # Get first image (original image)
        aberrated_image = images[0,:,:]
        # Get phase diversity images
        phase_diversity_images = images[1:,:,:]

        # return gt_image, aberrated_image, phase_diversity_images
        return gt_image, aberrated_image, phase_diversity_images

    def plot(self):
        # Plot GT image, aberrated image and phase diversity images (in subplots next to each other in one figure)
        # Use gridspec to create subplots
        fig = plt.figure(figsize=(32, 6))
        gs = gridspec.GridSpec(1, self.get_number_of_phase_diversities()+2)
        # Plot GT image (unaberrated image)
        ax = fig.add_subplot(gs[0, 0])
        self.image_subplot(ax, self.gt_image, 'Unaberrated image (full FOV)')
        # Plot aberrated image
        ax = fig.add_subplot(gs[0, 1])
        self.image_subplot(ax, self.aberrated_image, 'Aberrated image')
        # Plot phase diversity images
        for i in range(self.get_number_of_phase_diversities()):
            ax = fig.add_subplot(gs[0, i+2])
            self.image_subplot(ax, self.phase_diversity_images[i,:,:], f'Phase diversity image {i+1}')

    @staticmethod
    def image_subplot(ax, image, title, cmap='gray'):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

class MockData(ExperimentalDataset):
    def __init__(self, imgs, phase_aberration_coeffs, phase_diversity_coeffs,
                 iteration_number=1, microscope_parameters=MicroscopeParameters(1.2, 0.532*u.um, 0.104*u.um)):
        self.iteration_number = iteration_number
        self.microscope_parameters = microscope_parameters

        self.gt_image = ob
        self.gt_phase_aberration_coeffs = phase_aberration_coeffs
        self.phase_diversities_coeffs = phase_diversity_coeffs

        self.aberrated_image = imgs[0]
        self.phase_diversity_images[1:]


if __name__ == '__main__':
    data_dir = '/home/joren/documents/adaptive_optics/data/Datasets/AO/230921 AO0057 U2OS_Cell/'
    iteration_number = 1
    dataset = ExperimentalDataset(data_dir, iteration_number)
    print(dataset.gt_image.shape)
    dataset.plot()
    plt.show()
    print('Done.')
