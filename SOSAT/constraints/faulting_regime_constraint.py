import numpy as np
import numpy.ma as ma
from scipy.stats import rv_continuous

from .constraint_base import StressConstraint


def logistic(x, k, xo):
    return 1.0 / (1.0 + np.exp(-k * (x - xo)))


def int_logist(k, xo):
    return 2.0 - np.log(1.0 + np.exp(-k * (-xo - 1.0))) / k \
        + np.log(1.0 + np.exp(-k * (-xo + 1.0))) / k


def int_logist_a(a, k, xo):
    return a + 1 + np.log(np.exp(k * (xo - a)) + 1) / k \
        - np.log(np.exp(k * (xo + 1)) + 1) / k


class SU_gen(rv_continuous):
    """
    A class used to generate smoothed semi-uniform probability
    distributions for the scalar regime parameter. Most commonly this
    is used to define a probability distribution for the three
    Andersonian faulting regimes. A logistic function is used to smooth
    the transition between the different faulting regimes.

    Parameters
    ----------
    w_NF : float
        the weight assigned to stress states for which the scalar regime
        parameter is less than k2; with the default value of k2 this
        corresponds to normal faulting stress states; the
        absolute value only matters relative to the weight assigned to
        the other two faulting regimes

    w_SS : float
        the weight assigned to stress states for which the scalar
        regime parameter is between k1 and k2; with the default values
        of k1 and k2 this corresponds to strike-slip faulting stress
        states; the absolute value only matters relative to the weight
        assigned to the other two faulting regimes

    w_TF : float
        the weight assigned to stress states for which the scalar
        regime parameter is greater than k1; with the default value
        of k1 this corresponds to thrust faulting stress states; the
        absolute value only matters relative to the weight assigned to
        the other two faulting regimes

    theta1 : float, optional
        the first cutoff angle; defaults to a value of sqrt(2)/2,
        which is the value of the scalar regime parameter that marks
        the transition from thrust faulting to strike-slip faulting
        regimes

    k1 : float, optional
        a parameter controlling the width of the sigmoid transition at
        theta1. Larger numbers yield a more gradual transition. Default
        value is 300

    theta2 : float
        the second cutoff angle; default value is -sqrt(2)/2, which is
        the value of the scalar regime parameter that makrs the
        transition from strike-slip faulting to normal faulting.

    k2 : foat, opptional
        a parameter controlling the width of the sigmoid
        transition at theta2. Larger numbers yield a more
        gradual transition. Default value is 300

    Notes
    -----
    The methods of `scipy.stats.rv_continuous` that are overridden are
    private methods. The corresponding public methods check their
    arguments before calling the private methods.

    """
    def _argcheck(self, w_NF, w_SS, w_TF,
                  theta1, k1,
                  theta2, k2):
        """
        Function used to verify that arguments are valid. Had to
        override because the base clase version does not allow
        negative parameters, but they are required for this case.
        """

        valid = True
        if np.any(np.array([theta1, theta2]) < -1.0) or \
                            np.any(np.array([theta1, theta2]) > 1.0):
            valid = False
        elif np.any(np.array([k1, k2]) < 0.0):
            valid = False
        return valid

    def _pdf(self, x, w_NF, w_SS, w_TF,
             theta1, k1,
             theta2, k2):
        """
        Probability density function evaluated at x
        """
        nom = (w_NF - w_SS) * int_logist(k1, theta1) \
            + (w_SS - w_TF) * int_logist(k2, theta2) \
            + w_TF * 2.0
        func = (w_NF - w_SS) * logistic(x, k1, theta1) \
             + (w_SS - w_TF) * logistic(x, k2, theta2) \
             + w_TF
        return func / nom

    def _cdf(self, x, w_NF, w_SS, w_TF,
             theta1, k1,
             theta2, k2):
        """
        Cumulative distribution function evaluated at x
        """
        nom = (w_NF - w_SS) * int_logist(k1, theta1) \
            + (w_SS - w_TF) * int_logist(k2, theta2) \
            + w_TF * 2.0
        cdf = (w_NF - w_SS) * int_logist_a(x, k1, theta1) \
            + (w_SS - w_TF) * int_logist_a(x, k2, theta2) \
            + w_TF * (x + 1.0)
        return cdf / nom


SU = SU_gen(name='SU', a=-1.0, b=1.0)


class FaultingRegimeConstraint(StressConstraint):
    """
    A class that constrains the state of stress based on knowledge or
    assumptions about the relative probability of the three Andersonian
    faulting regimes.

    Parameters
    ----------
    regime_dist :  subclass of `scipy.stats.rv_continuous`
        The probability density function for the scalar regime
        parameter. See `StressState.regime` for details on the
        definition of the scalar regime parameter. Usually the
        and instance of `SU` is provided to apply a uniform
        probability to each Andersonian faulting regime with
        logistic functions to smooth the transition between
        each regime. However, any subclass of
        `scipy.stats.rv_continuous` defined from -1.0 to 1.0
        can be used.
    """
    def __init__(self,
                 regime_dist):
        """
        Constructor method
        """
        self._regime_dist = regime_dist

    def loglikelihood(self,
                      ss):
        regime = ss.regime()
        with np.errstate(divide='ignore'):
            loglikelihood = np.log(self._regime_dist.pdf(regime))
        return loglikelihood
