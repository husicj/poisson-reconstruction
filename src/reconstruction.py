import sys

import matplotlib.pyplot as plt
import numpy as np

from aberration import ZernikeAberration
from diversity_set import DiversitySet
from fast_fft import Fast_FFTs
from image import DataImage, MicroscopeImage

from bin.mem_profile import profile, memory_snapshot

class PoissonReconstruction:
    """
    This class serves as a data structure for creating and storing an image
    reconstrunction from a set of phase diversity images using an algorithm
    that assumes Poisson-like noise when calculating the maximum likelihood
    estimate of the object being imaged.

    It has the following attributes:

    aberration
        An instance of aberration.ZernikeAberration containing the maximum
        likelihood estimation of the unknown aberration present in the
        phase diversity images, such as would be needed to apply optical
        corrections in the microscope with the use of a deformable mirror or
        other types of adaptive optical component.

    break_condition_met
        A boolean that stores whether some break condition other than maximum
        iteration count has been reached. This is potentially updated during
        the execution of an iteration of the estimation algorithm, and is used
        to determine whether to continue to break out of the execution loop of
        the run method before the maximum number of iterations. It is then
        reset to False.

    center_coordinate
        The array coordinate for the center of square images of size self.size.
        This is a two component tuple, each the value of the nearest integer to
        half of self.size.

    diversity_set
        An instance of diversity_set.DiversitySet, this attribute contains the
        set of phase diversity images that are used as the basis of the
        reconstruction. Both the images themselves and the associated phase
        diversity aberrations (along with the relevant microscope parameters)
        are represented as instances of the image.MicroscopeImage class, found
        in the list diversity_set.images.

    image
        This is an instance of image.DataImage, and is a representation of the
        maximum likelihood image reconstruction of the object based on the
        provided set of diversity images after self.iteration_count iterations.

    iteration_count
        An integer that tracks the number of iterations of the reconstruction
        algorithm have been run.

    iteration_info
        A dictionary containing a number of additional supplementary and
        debugging data about the reconstruction iterations of a given instance
        of this class.

    size
        The size of the data images, and therefore of the image that is to be
        reconstructed. This is the full length of the side of the square images.

    step_size
        The starting step size to be used the next time the single_step()
        method is called. During the line search portion of the algoritm, this
        value is reduced by a factor of step_size_reduction_factor until a step
        of this size in the direction of the gradient results in an improved
        cost for the image estimate.

    step_size_reduction_factor
        The factor used the reduce the value of step_size when a step along
        the direction of the gradient overshoots an extremum resulting in a
        increase rather than reduction in the value of the cost function.

    And the following methods:

    run
        Runs repeated iterations of the maximum likelihood estimation algorithm
        and updates the corresponding attributes of the instance to reflect the
        estimates of the true image and unknown aberration present in the
        set of diversity images.
    
    single_step
        Runs a single iteration of the maximum likelihood estimation algorithm.
        This is used to implement the run method, but can also be used in a
        variety of situations independently, such as for debubgging.

    """
    
    def __init__(self,
                 diversity_set: DiversitySet,
                 estimated_coefficients_count: int = 21,
                 ffts: Fast_FFTs = None
                 ) -> None:
        self.aberration = None
        self.break_condition_met = False
        self.diversity_set = diversity_set
        self.size = diversity_set.images[0].shape[0]
        self.center_coordinate = (self.size//2, self.size//2)
        if ffts is not None:
            self.ffts = ffts
        else:
            self.ffts = Fast_FFTs(self.size, 1)
        self.image = MicroscopeImage.blank(self.size,
                                           self.ffts,
                                           diversity_set.microscope_parameters,
                                           self.aberration)
        self.iteration_count = 0
        self.iteration_info = {'cost': [-np.inf]}
        # self.step_size = 3e4
        self.step_size = 1
        self.step_size_reduction_factor =0.9
        # we start with a small search vector so that we can calculate an
        # initial cost to compare against in the first iteration
        self.search_direction_vector = 0.01 * np.ones((estimated_coefficients_count)) / self.step_size

    def run(self, max_iterations: int = 1000) -> None:
            """Iteratively runs the phase reconstruction algorithm until either
            self.iteration_count = max_iterations or break_condition_met
            becomes true."""

            while self.iteration_count < max_iterations:
                if self.break_condition_met:
                    self.break_condition_met = False
                    break
                self.single_step()

    def single_step(self) -> None:
        """Runs a single iteration of the phase reconstruction algorithm."""

        print(f"Iteration {self.iteration_count}")
        print(f"{self.search_direction_vector=}")
        cost = self._line_search()
        print(f"{self.diversity_set.ground_truth_aberration - self.aberration.coefficients=}")
        self._update_object_estimate_and_search_direction()
        # self.image.show()
        self.iteration_info['cost'].append(cost)
        self.iteration_count += 1

    def _line_search(self,
                     max_linesearch_iterations: int = 10
                     ) -> float:
        """Searches for an improvement of cost function within one dimension of
        the search space determined by self.search_direction_vector."""

        aberration_set = []
        for aberration in self.diversity_set.aberrations():
            aberration_set.append(self.aberration * aberration)
        for _ in range(max_linesearch_iterations):
            test_cost = 0
            for i, aberration in enumerate(aberration_set):
                step = self.step_size * self.search_direction_vector
                # TODO confirm that the following - sign is correct
                # test_coefficients = aberration.coefficients - step
                # TODO remove the following line
                test_coefficients = self.diversity_set.ground_truth_aberration
                test_aberration = ZernikeAberration(test_coefficients,
                                                    self.size,
                                                    self.ffts)
                # TODO the following line seems to be the main slowdown
                test_estimate = test_aberration.apply(self.image, True)
                test_cost += (self.diversity_set.images[i] *
                              np.log(test_estimate) - test_estimate).mean()[()]
            print(f"after line search iteration: {test_cost.real=}, {self.step_size=}")
            print(f"{test_aberration.coefficients=}")
            if test_cost.real > self.iteration_info['cost'][-1]:
                # Improvement over the previous iteration
                break
            else:
                self.step_size *= self.step_size_reduction_factor
        self.aberration = test_aberration
        return test_cost

    def _update_object_estimate_and_search_direction(self) -> None:
        """Used the aberration estimate in self.aberration to create an updated
        estimate of the object captured in self.diversity_set, and then update
        the search direction that can be used in the following iteration."""

        # Q is an intermediate term used in the updating of the object estimate
        # representing d_k(x) / (f(x) * s_k(x)), where d_k are the diversity
        # images, f(x) is the estimation of the true object, and s_k are the
        # point spread functions of the applied aberrations for diveristy image
        # k along with the estimated unknown aberration. Thus this ratio
        # represents the discrepancy of the estimate from the captured images.
        update_factor = DataImage.blank(self.size, self.ffts)#.view(dtype='complex128')
        coefficient_space_gradient = np.zeros(len(self.search_direction_vector))
        normalization_factor = 0
        for k in range(self.diversity_set.image_count):
            aberration_k = self.aberration * self.diversity_set.aberrations()[k]
            psf = aberration_k.psf(self.diversity_set.microscope_parameters)
            q = (self.diversity_set.images[k] /
                 (psf.fourier_transform * self.image.fft(self.ffts))
                 .fft(self.ffts))
            Q = q.fft(self.ffts) 
            update_factor_term_transform = np.conj(psf).fft(self.ffts) * Q
            # TODO check if this ignoring of imaginary components makes sense
            update_factor += update_factor_term_transform.fft(self.ffts).real
            normalization_factor += psf[self.center_coordinate]

            # the loss function gradient with respect to Zernike coefficients
            # is broken up here to help with readability
            temp1 = (np.conj(aberration_k.gpf(self.image.microscope_parameters)
                             .fft(self.ffts)) *
                     (Q * np.flip(self.image).fft(self.ffts))
                     .fft(self.ffts, force_inverse=True))
            temp2 = np.imag(aberration_k.gpf(self.image.microscope_parameters)
                            * temp1)
            for noll_index in range(len(coefficient_space_gradient)):
                zern = aberration_k.zernike_pixel_array(noll_index)
                coefficient_space_gradient[noll_index] += -2 * np.sum(zern * temp2)

        self.image *= update_factor / normalization_factor
        # self.search_direction_vector = -1 * coefficient_space_gradient / np.linalg.norm(coefficient_space_gradient)
        self.search_direction_vector = -1 * coefficient_space_gradient

    def __sizeof__(self):
        size = 0
        for attribute in dir(self):
            if isinstance(attribute, np.ndarray):
                size += sys.getsizeof(attribute.base)
            else:
                size += sys.getsizeof(attribute)
        return size
 

if __name__ == "__main__":
    # TODO change the path variable to be supplied by cl argument
    path = 'data_dir'
    diversity_set = DiversitySet.load_with_data_loader(path)
    recon = PoissonReconstruction(diversity_set)
    diversity_set.show()
    recon.run(max_iterations=15)
    recon.image.show()
