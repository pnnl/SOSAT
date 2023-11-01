import numpy as np
from scipy.stats import lognorm
from scipy.stats import beta
from scipy.stats import uniform
import matplotlib.pyplot as plt
import pint

from ..samplers import RejectionSampler

units = pint.UnitRegistry()


class HydraulicFracturing:
    """
    A class to evaluate the risk of hydraulic fracturing at a range
     of pore pressures.

    Parameters
    ----------
    stress_state : A `SOSAT.StressState` object
        The stress state to use to make the evaluation
    dPmax : float
        The maximum increase in pressure to consider
    gamma_dist : an object derived from `scipy.stats.rv_continuous`
        A probability distribution for the stress path coefficient
    T_dist : an object derived from `scipy.stats.rv_continuous`
        A probability distribution for the uniaxial tensile strength
        of the intact rock.
    """

    def __init__(self,
                 ss,
                 dPmax,
                 gamma_dist,
                 T_dist=None,
                 T_unit="MPa"):
        """
        Constructor method
        """
        self.dPmax = dPmax
        self.ss = ss
        self.gamma_dist = gamma_dist
        self.T_dist = T_dist
        self.T_unit = T_unit

    def EvaluatePfail(self, Npressures=20, Nsamples=1e6):
        """
        A method to evaluate the failure probability at pressures
        between the native pore pressure at self.dPmax.

        Parameters
        ----------
        Npressures : int
            The number of pressures between the native pore pressure
            and dPmax at which to evaluate the failure probability

        Nsamples : int
            The number of stress samples to use for the analysis

        Returns
        -------
        P, pfail : `numpy.ndarray`
            Array containing the pore pressures considered and the
            corresponding failure probabilities
        """
        Nsamples = int(Nsamples)
        Npressures = int(Npressures)

        # Generate samples of stress state magnitudes
        stress_sampler = RejectionSampler(self.ss)
        shmin, shmax, sv = stress_sampler.GenerateSamples(Nsamples)

        # Generate samples of stress path coefficients
        gamma = self.gamma_dist.rvs(Nsamples)

        # Generate samples of uniaxial tensile strength
        if self.T_dist is not None:
            T = self.T_dist.rvs(Nsamples) * units(self.T_unit)
            T = T.to(self.ss.stress_unit).magnitude

        Po = self.ss.pore_pressure
        dP_array = np.linspace(0.0, self.dPmax, Npressures)
        Pfail = np.zeros_like(dP_array)
        i = 0
        for dP in dP_array:
            # Evaluate effective stress using the stress path coefficient
            # for horizontal directions and assuming a constant total
            # stress in the vertical direction so that the effective
            # vertical stress decreases by the full pressure increment
            shmin_eff = shmin - Po + (gamma - 1.0) * dP
            shmax_eff = shmax - Po + (gamma - 1.0) * dP
            sv_eff = sv - Po - dP

            # Evaluate for failure (i.e. any tesnile effective stress)
            f_shmin_eff = np.zeros_like(shmin_eff)
            f_shmax_eff = np.zeros_like(shmax_eff)
            f_sv_eff = np.zeros_like(sv_eff)

            if self.T_dist is None:
                f_shmin_eff[shmin_eff < 0.0] = 1.0
                f_shmax_eff[shmax_eff < 0.0] = 1.0
                f_sv_eff[sv_eff < 0.0] = 1.0
            else:
                f_shmin_eff[shmin_eff < -T] = 1.0
                f_shmax_eff[shmax_eff < -T] = 1.0
                f_sv_eff[sv_eff < -T] = 1.0

            f = f_shmin_eff + f_shmax_eff + f_sv_eff

            Pfail[i] = np.sum(np.ones_like(f) * (f > 0.0)) \
                / np.sum(np.ones_like(f))

            i += 1

        return Po + dP_array, Pfail

    def PlotFailureProbability(self,
                               Npressures=20,
                               Nsamples=1e6,
                               figwidth=5.0):

        P, Pfail = self.EvaluatePfail(Npressures, Nsamples)

        fig = plt.figure(figsize=(figwidth, figwidth * 0.7))
        ax = fig.add_subplot(111)
        ax.plot(P, Pfail, "k")
        ax.set_xlabel("Pore Pressure")
        ax.set_ylabel("Probability of Hydraulic Fracturing")
        plt.tight_layout()

        return fig
