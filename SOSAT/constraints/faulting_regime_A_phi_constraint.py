import numpy as np
import numpy.ma as ma
from .constraint_base import StressConstraint


class FaultingRegimeAphiConstraint(StressConstraint):
    """
    A class that constrains the state of stress based on knowledge or
    assumptions about the probability of A_phi values.

    Parameters
    ----------
    A_phi_dist :  subclass of `scipy.stats.rv_continuous`
        The probability density function for the scalar A_phi value.
        See `StressState.A_phi_calculate` for details on the
        definition of the scalar regime parameter A_phi.
        Any subclass of `scipy.stats.rv_continuous`
        defined from 0 to 3 can be used.
        For example, the truncated normal distribution is used.
        from scipy.stats import truncnorm
        lower_bound = 0 (fixed)
        upper_bound = 3 (fixed)
        mean = 0.5 (changeable, mean of A_phi value at this region)
        variance = 0.08 (changable, variance of A_phi value at this region)
        std = np.sqrt(variance)
        rv = truncnorm((lower_bound-mean)/std, \
                       (upper_bound-mean)/std, mean, std)
    """

    def __init__(self, A_phi_dist):
        """
        Constructor method
        """
        self._A_phi_dist = A_phi_dist

    def loglikelihood(self, ss):
        A_phi = ss.A_phi_calculate()
        with np.errstate(divide='ignore'):
            loglikelihood = np.log(self._A_phi_dist.pdf(A_phi))
        return loglikelihood
