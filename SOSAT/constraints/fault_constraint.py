import numpy as np
import numpy.ma as ma
from scipy.stats import lognorm

from .constraint_base import StressConstraint


def critical_friction(sig1, sig3, pp):
    """
    A function that computes the critical friction coefficient
    that would induce slip for the given combination of minimum
    and maximum principal stress and pore pressure

    Parameters
    ----------
    sig1 : float or array_like
        The maximum (most compressive) principal stress. If `sig1` is
        array_like then `sig3` must be a scalar of type float
    sig3 : float or array_like
        The minimum (least compressive) principal stress.  If `sig3` is
        array_like then `sig1` must be a scalar of type float
    pp : float
        The pore pressure

    Returns
    -------
    mu_c : float or array_like
        The critical friction coefficient. If either sig1 or sig3 are
        array_like then the result will be an array containing the
        critical friction coefficient for each entry in the array.
    """

    arg = (sig1 - sig3) / (sig1 + sig3 - 2.0 * pp)
    phic = ma.arcsin(arg)
    muc = ma.tan(phic)
    return muc


class FaultConstraint(StressConstraint):
    """
    A class used to constrain the state of stress by the existence
    of frictional fault and fractures.

    Attributes
    ----------
    friction_dist : subclass of scipy.stats.rv_continuous
        The probability density function for the fault friction
        coefficient

    Parameters
    ----------
    friction_dist : subclass of scipy.stats.rv_continuous, optional
        a probability distribution function from the fault friction
        coefficient. If not provided the default is a lognormal
        distribution with s=0.15 and scale=0.7 passed into
        scipy.stats.lognorm
    """

    def __init__(self,
                 friction_dist=None):
        """
        Constructor method
        """
        if friction_dist is None:
            self.friction_dist = lognorm(scale=0.7,
                                         s=0.15)
        else:
            self.friction_dist = friction_dist

    def loglikelihood(self,
                      ss):
        """
        Computes the likelihood of each stress state

        Parameters
        ----------
        ss: `SOSAT.StressState` object
            StressState object containing the stress states
            over which the likelihood is to be evaluated

        Returns
        -------
        Numpy MaskedArray
            The returned object is a Numpy MaskedArray containing the
            likelihood for each stress `ss`. The returned array is
            masked identically to `ss.shmin_grid`
        """
        NFregime = np.sqrt(2.0) * 0.5
        TFregime = -np.sqrt(2.0) * 0.5
        regime = ss.regime()

        NF = regime > NFregime
        TF = regime < TFregime
        SS = (~NF) & (~TF)
        muc = ma.zeros(np.shape(ss.shmin_grid))
        muc.mask = ss.shmin_grid.mask
        muc[NF] = critical_friction(sig1=ss.vertical_stress,
                                    sig3=ss.shmin_grid[NF],
                                    pp=ss.pore_pressure)
        muc[SS] = critical_friction(sig1=ss.shmax_grid[SS],
                                    sig3=ss.shmin_grid[SS],
                                    pp=ss.pore_pressure)
        muc[TF] = critical_friction(sig1=ss.shmax_grid[TF],
                                    sig3=ss.vertical_stress,
                                    pp=ss.pore_pressure)

        log_likelihood = self.friction_dist.logsf(muc)

        return log_likelihood
