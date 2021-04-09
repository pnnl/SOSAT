import numpy as np
import numpy.ma as ma
from scipy.stats import lognorm


def critical_friction(sig1, sig3, pp):
    """A function that computes the critical friction coefficient
    that would induce slip for the given combination of minimum
    and maximum principal stress and pore pressure
    """

    arg = (sig1 - sig3) / (sig1 + sig3 - 2.0 * pp)
    phic = ma.arcsin(arg)
    return ma.tan(phic)


class FaultConstraint:
    """A class used to constrain the state of stress by the existence
    of frictional fault and fractures.

    :param friction_dist: a probability distribution function
    :type friction_dist: any object that inherits from
         scipy.stats.rv_continuous
    :param min_friction: minimum allowable friction coefficient
         (optional, default=0.01)
    :type min_friction: float
    :param max_friction: maximum allowable friction coefficient
         (optional, default=1.0)
    :type max_friction: float

    :param
    """

    def __init__(self,
                 friction_dist=None,
                 min_friction=0.01,
                 max_friction=1.0):
        """Constructor method
        """
        if friction_dist is None:
            self.friction_dist = lognorm(scale=0.7,
                                         s=0.15)
        else:
            self.friction_dist = friction_dist
        self.min_friction = min_friction
        self.max_friction = max_friction

    def likelihood(self,
                   ss):
        """ Computes the likelihood of each stress state
        :param ss: StressState object
        :type ss: StressState object containing the stress states
            to be evaluated

        :return: An array containing the likelihood for each stress
          state included in ``ss``
        :rtype: array of same shape as stress arrays in ``ss``, which
            is currently a masked meshgrid array containing the bins for
            the minimum and maximum horizontal stress
        """
        NFregime = np.sqrt(2.0) * 0.5
        TFregime = -np.sqrt(2.0) * 0.5
        regime = ss.regime()

        NF = regime > NFregime
        TF = regime < TFregime
        SS = (~NF) & (~TF)
        muc = ma.zeros(np.shape(ss.shmax_grid))
        muc[NF] = critical_friction(sig1=ss.vertical_stress,
                                    sig3=ss.shmin_grid[NF],
                                    pp=ss.pore_pressure)
        muc[SS] = critical_friction(sig1=ss.shmax_grid[SS],
                                    sig3=ss.shmin_grid[SS],
                                    pp=ss.pore_pressure)
        muc[TF] = critical_friction(sig1=ss.shmax_grid[TF],
                                    sig3=ss.vertical_stress,
                                    pp=ss.pore_pressure)
        return self.friction_dist.sf(muc)
