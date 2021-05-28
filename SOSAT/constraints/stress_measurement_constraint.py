import numpy as np
from numpy import ma
import pint

from .constraint_base import StressConstraint


class StressMeasurement(StressConstraint):
    """
    A class to incorporate mini-frac or extended leakoff test (XLOT)
    data to constrain the state of stress

    Attributes
    ----------
    No public attributes

    Parameters
    ----------
    shmin_dist : subclass of `scipy.stats.rv_continuous`
        The probability distribution for the measured minimum
        horizontal stress

    Notes
    -----
    This assumes that the minimum principal stress, which is measured
    by mini-frac or XLOT tests is horizontal. This makes this class
    only applicable to normal faulting or strike-slip stress
    environments
    """

    def __init__(self,
                 shmin_dist):
        """
        Constructor method
        """
        self._shmin_dist = shmin_dist

    def loglikelihood(self, ss):
        """
        Compute the likelihood of each stress state in `ss` based on
        the measured stress

        Parameters
        ----------
        ss : `SOSAT.StressState` object
            The stress states on which to evaluate the likelihood
        """

        return self._shmin_dist.logpdf(ss.shmin_grid)
