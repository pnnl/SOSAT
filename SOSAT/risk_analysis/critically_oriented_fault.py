import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

from ..samplers import RejectionSampler


def fmt(x, pos):
    """
    A utility function to improve the formatting of
    plot labels
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


class CriticalFaultActivation:
    """
    A class to evaluate the risk of activation of a critically oriented
    fault at a range of pore pressures.

    Parameters
    ----------
    stress_state : A `SOSAT.StressState` object
        The stress state to use to make the evaluation
    dPmax : float
        The maximum increase in pressure to consider
    gamma_dist : an object derived from `scipy.stats.rv_continuous`
        A probability distribution for the stress path coefficient
    mu_dist : an object derived from `scipy.stats.rv_continuous`,
              optional
        A probability distribution for the fault friction coefficient;
        defaults to a lognormal distribution with s=0.14 and scale=0.7
        passed into `scipy.stats.lognorm`
    """

    def __init__(self,
                 ss,
                 dPmax,
                 gamma_dist,
                 mu_dist=None):
        """
        Constructor method
        """
        self.dPmax = dPmax
        self.ss = ss
        self.gamma_dist = gamma_dist
        if mu_dist is None:
            self.mu_dist = lognorm(scale=0.7,
                                   s=0.15)
        else:
            self.mu_dist = mu_dist

    def SampleStressPoints(self, Nsamples=1e6):
        """
        A method to sample three principal stress points
        from the joint stress posterior distribution

        Parameters
        ----------
        Nsamples : int
            The number of stress samples to use for the analysis

        Returns
        -------
        shmin, shmax, sv: arrays containing samples of the three
        principal stress

        """
        Nsamples = int(Nsamples)
        # generate samples of stress state
        stress_sampler = RejectionSampler(self.ss)
        shmin, shmax, sv = stress_sampler.GenerateSamples(Nsamples)
        return shmin, shmax, sv

    def EvaluatePfail(self, Npressures=20, Nsamples=1e6,
                      shmin=None, shmax=None, sv=None):
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

        shmin, shmax, sv: arrays containing samples of the three
        principal stress; default to be None; They can be calculated
        using self.SampleStressPoints(Nsamples=1e6)

        Returns
        -------
        P, pfail : `numpy.ndarray`
            Array containing the pore pressures considered and the
            corresponding failure probabilities
        """
        Nsamples = int(Nsamples)
        Npressures = int(Npressures)

        # generate samples of stress state if there is no stress inputs
        if shmin is None or shmax is None or sv is None:
            stress_sampler = RejectionSampler(self.ss)
            shmin, shmax, sv = stress_sampler.GenerateSamples(Nsamples)
        else:
            # make sure the Nsamples equals to the given length of shmin
            Nsamples = len(sv)

        gamma = self.gamma_dist.rvs(Nsamples)
        mu = self.mu_dist.rvs(Nsamples)

        Po = self.ss.pore_pressure
        dP_array = np.linspace(0.0, self.dPmax, Npressures)
        Pfail = np.zeros_like(dP_array)
        i = 0
        for dP in dP_array:
            # evaluate effective stress using the stress path coefficient
            # for horizontal directions and assuming a constant total
            # stress in the vertical direction so that the effective
            # vertical stress decreases by the full pressure increment
            shmin_eff = shmin - Po + (gamma - 1.0) * dP
            shmax_eff = shmax - Po + (gamma - 1.0) * dP
            sv_eff = sv - Po - dP

            # evaluate faulting regime at perturbed state
            NF = sv_eff > shmax_eff
            TF = sv_eff < shmin_eff
            SS = ~NF & ~TF

            S1_eff = np.zeros(Nsamples, dtype=np.float64)
            S3_eff = np.zeros(Nsamples, dtype=np.float64)

            S1_eff[NF] = sv_eff[NF]
            S3_eff[NF] = shmin_eff[NF]

            S1_eff[SS] = shmax_eff[SS]
            S3_eff[SS] = shmin_eff[SS]

            S1_eff[TF] = shmax_eff[TF]
            S3_eff[TF] = sv_eff[TF]

            phi = np.arctan(mu)
            q = 0.5 * (S1_eff - S3_eff)
            p = 0.5 * (S1_eff + S3_eff)
            f = q - p * np.sin(phi)

            Pfail[i] = np.sum(np.ones_like(f) * (f > 0.0)) \
                       / np.sum(np.ones(Nsamples))

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
        ax.set_xlabel("Pore Pressure (" + self.ss.stress_unit + ")")
        ax.set_ylabel("Probability of Fault Activation")
        plt.tight_layout()

        return fig
