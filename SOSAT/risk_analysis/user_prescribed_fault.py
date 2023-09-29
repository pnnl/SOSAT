import numpy as np
from scipy.stats import lognorm
from scipy.stats import beta
from scipy.stats import uniform
import matplotlib.pyplot as plt

from ..samplers import RejectionSampler


def rand_vMF(vec, K, Nsamples):
    """
    Random sampling from von Mises - Fisher distribution with mean
    vector (vec) and concentration K. Based on method proposed by Pinzon and
    Jung (2023).

    Parameters
    ----------
    vec : list of length 3
        The components of the mean vector.
    K : float
        The concentration parameter.
    """

    # Check mean vector is a unit vector
    if np.round(np.linalg.norm(vec), 6) != 1.0:
        error_message = "Vector supplied to the von Mises-Fisher sampler" + \
            " is not a unit vector."
        raise ValueError(error_message)
    # Check mean vector has length 3
    if np.shape(vec) != (1, 3):
        error_message = "Vector supplied to the von Mises-Fisher sampler" + \
            " is not three-dimensional."

    Nsamples = int(Nsamples)
    vec = np.array(vec)

    # Samples perpendicular to mean vector
    # Random vectors
    Z = np.random.normal(0, 1, (Nsamples, 3))
    # Normalize to unit vectors
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    # Ensure they are perpendicular to mean vector
    Z = Z - (Z @ vec[:, np.newaxis]) * vec[np.newaxis, :]
    # Normalize to unit vectors
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)

    # Sample theta angles (cos and sin)
    theta_cos = vMF_angle(K, Nsamples)
    theta_sin = np.sqrt(1 - theta_cos**2)

    X = Z * theta_sin[:, np.newaxis] + theta_cos[:, np.newaxis]\
        * vec[np.newaxis, :]

    return X.reshape((Nsamples, 3))


def vMF_angle(K, Nsamples):
    """
    Create samples with density function given by
    p(t) = someConstant * (1-t**2) * exp(K *t)
    """

    t_i = np.sqrt(1 + (1 / K)**2) - 1 / K
    r_i = t_i
    logt_i = K * t_i + 2 * np.log(1 - r_i * t_i)
    count = 0
    results = []
    while count < Nsamples:
        m = min(Nsamples, int((Nsamples - count) * 1.5))
        t = np.random.beta(1, 1, m)
        t = 2 * t - 1
        t = (r_i + t) / (1 + r_i * t)
        log_acc = K * t + 2 * np.log(1 - r_i * t) - logt_i
        t = t[np.random.random(m) < np .exp(log_acc)]
        results.append(t)
        count += len(results[-1])

    return np.concatenate(results)[:Nsamples]


class UserPrescribedFaultActivation:
    """
    A class to evaluate the risk of activation of a fault with
    user-prescribed orientation at a range of pore pressures.

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
    ss_azi_m : float
        Mean value for the orientation (angle) of the maximum
        horizontal stress, with an angle of 0 degrees being North and
        increasing clockwise (i.e. E = 90 deg, S = 180 deg,
        W = 20 deg)
    ss_K : float
        K-value for Von Mises-Fisher distribution for the orientation
        of the maximum horizontal stress, with a larger number
        producing less uncertainty
    strike_m : float
        Mean value forr the orientation (angle) of the fault strike,
        with an angle of 0 degrees being North and increasing
        clockwise (i.e. E = 90 deg, S = 180 deg, W = 20 deg)
    dip_m : float
        Mean value for the orientation (angle) of the fault dip, with
        an angle of 0 degrees being horizontal and 90 degrees being
        vertical
    fault_K : float
        K-value for Von Mises-Fisher distribution for the orientation
        of the fault, with a larger number producing
        less uncertainty
    """

    def __init__(self,
                 ss,
                 dPmax,
                 gamma_dist,
                 ss_azi_m,
                 ss_K,
                 strike_m,
                 dip_m,
                 fault_K,
                 mu_dist=None):
        """
        Constructor method
        """
        self.dPmax = dPmax
        self.ss = ss
        self.gamma_dist = gamma_dist
        self.ss_azi_m = ss_azi_m
        self.ss_K = ss_K
        self.strike_m = strike_m
        self.dip_m = dip_m
        self.fault_K = fault_K
        if mu_dist is None:
            self.mu_dist = lognorm(scale=0.7,
                                   s=0.15)
        else:
            self.mu_dist = mu_dist

        if ss_azi_m < 0.0 or ss_azi_m > 360.0:
            error_message = 'Invalid value for the azimuth of the maximum' + \
                ' horizontal stress (ss_azi_m): ' + str(ss_azi_m) + \
                '. Expected value between 0 and 360.'
            raise ValueError(error_message)

        if strike_m < 0.0 or strike_m > 360.0:
            error_message = 'Invalid value for the fault strike angle' + \
                ' (strike_m): ' + str(strike_m) + \
                '. Expected value between 0' + ' and 360.'
            raise ValueError(error_message)

        if dip_m < 0.0 or dip_m > 90.0:
            error_message = 'Invalid value for the fault dip angle' + \
                ' (dip_m): ' + str(dip_m) + \
                '. Expected value between 0 and 90.'
            raise ValueError(error_message)

    def CreateOrientations(self, Nsamples=1e6):
        """
        A method to create the orientations of the stress state and fault,
        based on the user-supplied mean orientation (i.e. azimuth of maximum
        horizontal stress or the fault's strike and dip) and Von
        Mises-Fisher K-values.

        Parameters
        ----------
        Nsamples : int
            The number of samples of stress state and fault orientations
            to use

        Returns
        -------
        n_shmax, n_fault, n_fault_p, ss_azi, strike_fault, dip_fault :
        `numpy.ndarray`
            Arrays containing the normal vectors for shmax samples, normal
            vectors for the fault samples, the fault normal vectors in the
            principal coordinate system, azimuths for shmax samples, strike
            angles for fault samples, and dip angles
            for fault samples
        """
        Nsamples = int(Nsamples)

        # Generate samples of stress state orientations:
        # n_shmax_m = [x1, x2, x3] = [x_north, x_east, x_down]
        n_shmax_m = [np.cos(np.radians(self.ss_azi_m)),
                     np.sin(np.radians(self.ss_azi_m)), 0.0]
        n_shmax = rand_vMF(n_shmax_m, self.ss_K, Nsamples)
        # Project n_shmax back into horizontal plane and extend to unity
        n_shmax = np.array([[x[0], x[1], 0.0]
                            / (np.sqrt(x[0]**2.0 + x[1]**2.0))
                            for x in n_shmax])
        # Calculate azimuth of sample stress state orientations from normals
        ss_azi = np.degrees(np.arccos(n_shmax[:, 0]))
        # Stress azimuth must be corrected if n_shmaz[:, 1] < 0
        ss_azi[n_shmax[:, 1] < 0.0] = 360.0 - ss_azi[n_shmax[:, 1] < 0.0]
        # Correct stress azimuth out of bounds
        ss_azi[ss_azi > 360.0] -= 360.0
        ss_azi[ss_azi < 0.0] += 360.0

        # Generate samples of fault orientations:
        # n_fault_m = [x1, x2, x3] = [x_north, x_east, x_down]
        n_fault_m = [-np.sin(np.radians(self.dip_m))
                     * np.sin(np.radians(self.strike_m)),
                     np.sin(np.radians(self.dip_m))
                     * np.cos(np.radians(self.strike_m)),
                     -np.cos(np.radians(self.dip_m))]
        n_fault = rand_vMF(n_fault_m, self.fault_K, Nsamples)
        # Calculate sample strike and dip angles from normals
        # Avoid np.arccos() argument greater than 1 (caused by rounding errors)
        dip_fault = np.degrees(np.arccos(-n_fault[:, 2]))
        strike_fault = np.zeros_like(dip_fault)
        arg_mask = np.array([n_fault[:, 1]
                            / np.sin(np.radians(dip_fault)) < 1.0])[0, :]
        strike_fault[arg_mask] = np.degrees(np.arccos(n_fault[:, 1][arg_mask]
                                            / np.sin(np.radians(dip_fault
                                                                [arg_mask]))))
        # Strike angles must be corrected if n_fault[:, 0] > 0
        strike_fault[n_fault[:, 0] > 0.0] = 360.0 \
            - strike_fault[n_fault[:, 0] > 0.0]
        # Correct strike and dip if dip > 90 degrees
        strike_fault[dip_fault > 90.0] += 180.0
        dip_fault[dip_fault > 90.0] = 180.0 - dip_fault[dip_fault > 90.0]
        # Correct strike and dip if dip < 0 degrees
        strike_fault[dip_fault < 0.0] += 180.0
        dip_fault[dip_fault < 0.0] *= -1
        # Correct strike angles out of bounds
        strike_fault[strike_fault > 360.0] -= 360.0
        strike_fault[strike_fault < 0.0] += 360.0

        # Calculate the fault azimuth:
        fault_azi = np.degrees(np.arcsin(n_fault[:, 1]
                                         / np.sqrt(n_fault[:, 0]**2.0
                                                   + n_fault[:, 1]**2.0)))
        # Fault azimuth must be corrected if n_fault[:, 0] < 0
        fault_azi[n_fault[:, 0] < 0.0] = 180.0 - fault_azi[n_fault[:, 0] < 0.0]

        # Project fault normals into the principal coordinate system:
        n_fault_p = np.ones_like(n_fault)
        n_fault_p[:, 0] = np.sqrt(n_fault[:, 0]**2.0 + n_fault[:, 1]**2.0) * \
            np.cos(np.radians(fault_azi) - np.radians(ss_azi))
        n_fault_p[:, 1] = np.sqrt(n_fault[:, 0]**2.0 + n_fault[:, 1]**2.0) * \
            np.sin(np.radians(fault_azi) - np.radians(ss_azi))
        n_fault_p[:, 2] = n_fault[:, 2]

        return n_shmax, n_fault, n_fault_p, ss_azi, strike_fault, dip_fault

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

        # Generate samples of fault friction coefficient
        mu = self.mu_dist.rvs(Nsamples)

        # Create orientations for stress state and fault
        n_shmax, n_fault, n_fault_p, ss_azi, strike_fault, dip_fault = \
            self.CreateOrientations(Nsamples)

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

            # Calculate the effective normal and shear stress magnitudes
            sigma_n_eff = shmax_eff * n_fault_p[:, 0]**2.0 + \
                shmin_eff * n_fault_p[:, 1]**2.0 + \
                sv_eff * n_fault_p[:, 2]**2.0

            tau_1 = (shmin_eff - sv_eff) * n_fault_p[:, 1] * n_fault_p[:, 2]
            tau_2 = (sv_eff - shmax_eff) * n_fault_p[:, 2] * n_fault_p[:, 0]
            tau_3 = (shmax_eff - shmin_eff) * n_fault_p[:, 0] * n_fault_p[:, 1]

            tau = np.sqrt(tau_1**2.0 + tau_2**2.0 + tau_3**2.0)

            # Evaluate Mohr-Coulomb for failure
            f = mu * sigma_n_eff - tau

            Pfail[i] = np.sum(np.ones_like(f) * (f < 0.0)) \
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
        ax.set_ylabel("Probability of Fault Activation")
        plt.tight_layout()

        return fig

    def PlotStrikeOrientation(self,
                              figwidth=5.0):

        n_shmax, n_fault, n_fault_p, ss_azi, strike_fault, dip_fault = \
            self.CreateOrientations()

        fig = plt.figure(figsize=(figwidth, figwidth * 0.7))
        ax = fig.add_subplot(111)
        ax.hist(strike_fault, 100)
        ax.set_xlabel(r"Fault Strike [$^{\circ}$]")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 360)
        plt.tight_layout()

        return fig

    def PlotDipOrientation(self,
                           figwidth=5.0):

        n_shmax, n_fault, n_fault_p, ss_azi, strike_fault, dip_fault = \
            self.CreateOrientations()

        fig = plt.figure(figsize=(figwidth, figwidth * 0.7))
        ax = fig.add_subplot(111)
        ax.hist(dip_fault, 100)
        ax.set_xlabel(r"Fault Dip [$^{\circ}$]")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 90)
        plt.tight_layout()

        return fig

    def PlotStressOrientation(self,
                              figwidth=5.0):

        n_shmax, n_fault, n_fault_p, ss_azi, strike_fault, dip_fault = \
            self.CreateOrientations()

        fig = plt.figure(figsize=(figwidth, figwidth * 0.7))
        ax = fig.add_subplot(111)
        ax.hist(ss_azi, 100)
        ax.set_xlabel(r"Azimuth of Maximum Horizontal Stress [$^{\circ}$]")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 360)
        plt.tight_layout()

        return fig
