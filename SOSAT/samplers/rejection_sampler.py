from scipy.stats import randint, uniform
import numpy as np


"""
Function that determines the maximum value of the maximum horizontal
stress allowed by fault strength
"""


def sigHcritTF(sigv, pp, mu):
    return (np.sqrt(mu ** 2 + 1.0) + mu) ** 2 * (sigv - pp) + pp


"""
Function that determines the minimum value of the minimum horizontal
stress allowed by fault strength
"""


def sighcritNF(sigv, pp, mu):
    scrit = (sigv - pp) / (np.sqrt(mu ** 2 + 1.0) + mu) ** 2 + pp
    return scrit


class RejectionSampler:
    """
    A class used to generate samples of the posterior stress distribution
    using a rejection sampling approach

    Parameters
    ----------
    SS : `SOSAT.StressState` object
    """
    def __init__(self,
                 SS):
        self.SS = SS

    def GenerateSamples(self, Nsamples):
        """
        Method to evaluate samples

        Arguments
        ---------
        Nsamples: int
            Number of samples to generate

        Returns
        -------
        shmin, shmax, sv: arrays containing samples of the three
                          principal stress
        """

        # evaluate the poserior on the grid
        self.SS.evaluate_posterior()
        pmax = np.max(self.SS.posterior)

        # scale to 0-1
        psig = self.SS.posterior / pmax

        # find the size of the stress grid
        ngrid = np.shape(psig)[0]

        # now generate Nsamples random draws of the index of shmin and
        # separately for shmax.
        # use the randint distribution. When only given one argument
        # that is the upper bound and the lower bound is assumed to be
        # zero, which is want we want
        indx_gen = randint(0, ngrid)
        ugen = uniform()
        Naccepted = 0
        shmax_samples = []
        shmin_samples = []
        sv_samples = np.ones(Nsamples) * self.SS.vertical_stress
        while Naccepted < Nsamples:
            shmin_index = indx_gen.rvs(Nsamples)
            shmax_index = indx_gen.rvs(Nsamples)

            # find all samples where max<min and swap them
            swap = shmin_index > shmax_index
            jnk = shmin_index[swap]
            shmin_index[swap] = shmax_index[swap]
            shmax_index[swap] = jnk

            psamp = psig[shmin_index, shmax_index]
            shmax_t = self.SS.shmax_grid[shmin_index, shmax_index]
            shmin_t = self.SS.shmin_grid[shmin_index, shmax_index]

            u = ugen.rvs(Nsamples)

            # accept those where psamp > u
            accept = psamp > u

            shmin_samples = np.concatenate((shmin_samples, shmin_t[accept]))
            shmax_samples = np.concatenate((shmax_samples, shmax_t[accept]))
            Naccepted = len(shmin_samples)

        shmin_samples = shmin_samples[0:Nsamples]
        shmax_samples = shmax_samples[0:Nsamples]

        return shmin_samples, shmax_samples, sv_samples
