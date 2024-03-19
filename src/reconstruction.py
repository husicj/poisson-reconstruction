from aberration import ZernikeAberration
from diversity_set import DiversitySet
from image import DataImage

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
                 estimated_coefficients_count: int = 21
                 ) -> None:
        self.aberration = None
        self.break_condition_met = False
        self.diversity_set = diversity_set
        self.image = DataImage.blank()
        self.iteration_count = 0
        self.iteration_info = {'cost' = [-np.inf]}
        self.step_size = 3e4
        self.step_size_reduction_factor = 0.3
        self.search_direction_vector = np.zeros((estimated_coefficients_count))

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
        cost = self._line_search()
        self.iteration_count += 1

    def _line_search(self,
                     max_linesearch_iterations: int = 10
                     ) -> float:
        for i in range(max_linesearch_iterations):
            step = self.step_size * self.search_direction_vector
            test_coefficients = self.aberration.coefficients - step
            test_aberration = ZernikeAberration(test_coefficients)
            test_estimate = test_aberration.apply(self.image, True)
            test_cost = (np.array(self.diversity_set.images) *
                    np.log(test_estimate) - test_estimate).mean()
            if test_cost > self.iteration_info['cost'][-1]:
                # Improvement over the previous iteration
                break
            else:
                self.step_size *= step_size_reduction_factor
        self.aberration = test_aberration
        return test_cost
