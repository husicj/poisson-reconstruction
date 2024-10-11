import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        self.aberration_estimate = None
        self.break_condition_met = False
        self.diversity_set = diversity_set
        self.size = diversity_set.images[0].shape[0]
        self.center_coordinate = (self.size//2, self.size//2)
        if ffts is not None:
            self.ffts = ffts
        else:
            self.ffts = Fast_FFTs(self.size, 1)
        self.object_estimate = MicroscopeImage.black(self.size,
                                                     self.ffts,
                                                     diversity_set.microscope_parameters,
                                                     self.aberration_estimate)
        self.iteration_count = 0
        self.iteration_info = {'likelihood': [-np.inf]}
        self.step_size = 3e4
        # self.step_size = 1
        self.step_size_reduction_factor = 0.3 #0.9 # FIX: used .3 in iter_p
        # we start with a small search vector so that we can calculate an
        # initial cost to compare against in the first iteration
        self.search_direction_vector = 0.01 * np.ones((estimated_coefficients_count)) / self.step_size

    def run(self, max_iterations: int = 1000) -> None:
            """Iteratively runs the phase reconstruction algorithm until either
            self.iteration_count = max_iterations or break_condition_met
            becomes true."""

            with tqdm(total=max_iterations, desc="Iterations") as iteration_timer:
                while self.iteration_count < max_iterations:
                    if self.break_condition_met:
                        self.break_condition_met = False
                        break
                    self.single_step()
                    iteration_timer.update(1)

    def single_step(self) -> None:
        """Runs a single iteration of the phase reconstruction algorithm."""

        # print(f"Iteration {self.iteration_count}")
        # print(f"{self.search_direction_vector=}")
        cost = self._line_search()
        # print(f"{self.diversity_set.ground_truth_aberration - self.aberration.coefficients=}")
        self._update_object_estimate_and_search_direction()
        self.iteration_info['likelihood'].append(cost)
        self.iteration_count += 1

    def _line_search(self,
                     max_linesearch_iterations: int = 10
                     ) -> float:
        """Searches for an improvement of cost function within one dimension of
        the search space determined by self.search_direction_vector."""

        aberration_set = []
        for aberration in self.diversity_set.aberrations():
            aberration_set.append(self.aberration_estimate * aberration)

        with tqdm(total=max_linesearch_iterations, leave=False, desc="Line search", position=1) as linesearch_timer:
            for _ in range(max_linesearch_iterations):
                likelihood = 0
                for i, aberration in enumerate(aberration_set):
                    step = self.step_size * self.search_direction_vector
                    coefficients_estimate = aberration.coefficients + step
                    aberration_estimate = ZernikeAberration(coefficients_estimate, self.size, self.ffts)

                    # TODO the following line seems to be the main slowdown
                    image_estimates = aberration_estimate.apply(self.object_estimate, True)

                    # print(f"{test_aberration.coefficients=}")
                    # the [()] indexing removes the MicroscopeImage wrapper from
                    # the value, since np.mean() preserves object type here

                    likelihood += (self.diversity_set.images[i] * np.log(image_estimates) - image_estimates).sum()[()]

                linesearch_timer.set_postfix(
                    likelihood_real=f'{likelihood.real:.4e}',
                    step_size=f'{self.step_size:.4e}'
                )
                linesearch_timer.update(1)
                # print(f"after line search iteration: {test_cost.real=}, {self.step_size=}")

                if likelihood.real > self.iteration_info['likelihood'][-1]:
                    # Improvement over the previous iteration
                    break
                else:
                    self.step_size *= self.step_size_reduction_factor

        self.aberration_estimate = aberration_estimate
        return likelihood

    def _update_object_estimate_and_search_direction(self) -> None:
        """Used the aberration estimate in self.aberration to create an updated
        estimate of the object captured in self.diversity_set, and then update
        the search direction that can be used in the following iteration."""

        # Q is an intermediate term used in the updating of the object estimate
        # representing d_k(x) / (f(x) * s_k(x)), where d_k are the diversity
        # images, f(x) is the estimation of the true object, and s_k are the
        # point spread functions of the applied aberrations for diveristy image
        # k along with the estimated unknown aberration. Thus, this ratio
        # represents the discrepancy of the estimate from the captured images.

        self.object_estimate.fftshift()

        update_factor = DataImage.blank(self.size, self.ffts)
        coefficient_space_gradient = np.zeros(len(self.search_direction_vector))
        normalization_factor = 0
        gradient_inner_sum = 0

        for k in range(self.diversity_set.image_count):
            aberration_k = self.aberration_estimate + self.diversity_set.aberrations()[k]
            psf_k = aberration_k.psf(self.diversity_set.microscope_parameters) # s_k
            gpf_k = aberration_k.gpf(self.diversity_set.microscope_parameters) # H_k

            # Calculate "fraction term" required for both gradient and object update
            image_estimate_k_fft = psf_k.fourier_transform * self.object_estimate.fft(self.ffts)
            q = self.diversity_set.images[k] / image_estimate_k_fft.fft(self.ffts)
            Q = q.fft(self.ffts)

            # Object update factor
            update_factor_term_transform = np.flip(psf_k).fft(self.ffts) * Q # NOTE: could also replace flip in object space with conjugate in Fourier space
            update_factor += update_factor_term_transform.fft(self.ffts).real
            normalization_factor += np.sum(psf_k)

            # Coefficient gradient terms
            h_k_conj = np.conj(gpf_k.fft(self.ffts))
            inverse_fourier_term = ((h_k_conj * np.flip(self.object_estimate).fft(self.ffts) * Q)
                                    .fft(self.ffts, force_inverse=True))
            gradient_inner_sum += gpf_k * inverse_fourier_term

        ### Coefficient gradient calculation
        for noll_index in range(len(coefficient_space_gradient)):
            # TODO: check calculation of Zernikes! Normalization seems to be missing (both Zernike normalization and lambda/NA term)
            zern = self.aberration_estimate.zernike_pixel_array(noll_index)
            coefficient_space_gradient[noll_index] = -2 * np.sum(zern * np.imag(gradient_inner_sum)) # TODO: do we need to divide by size * size? would correspond to taking the mean

        ### Object update
        # Apply fftshift to the update factor to match the natural order of the image
        update_factor = update_factor.fftshift()
        self.object_estimate.fftshift()
        self.object_estimate = self.object_estimate * update_factor / normalization_factor

        ### Search direction update
        self.search_direction_vector = coefficient_space_gradient

    def __sizeof__(self):
        size = 0
        for attribute in dir(self):
            if isinstance(attribute, np.ndarray):
                size += sys.getsizeof(attribute.base)
            else:
                size += sys.getsizeof(attribute)
        return size
 

if __name__ == "__main__":
    # Load data
    # TODO change the path variable to be supplied by cl argument
    path = Path(__file__).parent / "../data_dir"
    diversity_set = DiversitySet.load_with_data_loader(path).crop(256)
    diversity_set.show()

    # Run reconstruction
    recon = PoissonReconstruction(diversity_set)
    recon.run(max_iterations=2)

    # Show results
    # Print with significant digits
    print(f"Ground truth aberration: {diversity_set.ground_truth_aberration}")
    print(f"Estimated aberration: {recon.aberration_estimate.coefficients}")
    recon.object_estimate.show()
    recon.aberration_estimate.get_phase(recon.object_estimate.microscope_parameters).show()