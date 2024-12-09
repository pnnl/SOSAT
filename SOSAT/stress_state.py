import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pint
import scipy.stats
from scipy.spatial import KDTree
units = pint.UnitRegistry()

"""
The stress_state module contains classes and functions that are used to
represent and plot the joint probability distribution for the minimum
and maximum horizontal stresses.
"""

gravity = 9.81 * units('m/s^2')


def fmt(x, pos):
    """
    A utility function to improve the formatting of
    plot labels
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


class StressState:
    """
    A class to contain all data necessary to define the probability
    distribution for all possible stress states at a given point in the
    subsurface.

    Attributes
    ----------
    depth : float
        The true vertical depth of the point being analyzed
    vertical_stress : float
        The total vertical stress, currently taken as deterministic
    pore_pressure : float
        The pore pressure, currently taken as deterministic
    shmin_grid : numpy MaskedArray object
        A 2D array containing the value of the minimum horizontal
        stress for each stress state considered. Values that are not
        admissible, such as where the minimum horizontal stress would
        be greater than the maximum horizontal stress are masked
    shmax_grid : numpy MaskedArray object
        A 2D array containing the value of the mmaximum horizontal
        stress for each stress state considered. Values that are not
        admissible, such as where the minimum horizontal stress would
        be greater than the maximum horizontal stress are masked
    stress_unit : str
        The unit used for stress
    depth_unit : str
        The unit used for vertical depth

    Parameters
    ----------
    depth : float
        the true vertical depth of the point being analyzed
    average_overburden_density : float
         average mass density of all overlying formations
    pore_pressure : float
        formation pore pressure
    depth_unit : str, optional
        unit of measurement for depth, see list of units in pint
        package documentation
    density_unit : str, optional
        unit of measurement for mass density, see list of units in
        pintpackage documentation
    pressure_unit : str, optional
        unit of measurement for pressure, see list of units in pint
        package documentation
    min_stress_ratio : float, optional
        Minimum stress included in the analysis expressed as a fraction
        of the vertical stress. This is not intended as a way to express
        a prior probability but is instead meant as a convenient way to
        set the bounds of stress considered in the analysis. It does
        effectively truncate the stress distribution at this value and
        so should be chose well outside of the zone of significant
        probability density. Default value is 0.4
    max_stress_ratio : float
        Maximum stress included in the analysis expressed as a fraction
        of the vertical stress. This is not intended as a way to express
        a prior probability but is instead meant as a convenient way to
        set the bounds of stress considered in the analysis. It does
        effectively truncate the stress distribution at this value and
        so should be chose well outside of the zone of significant
        probability density. Default value is 3.25
    nbins : int
        number of bins to use for the horizontal principal stresses.
        The same number is used for both horizontal principal stresses.
        Default value is 200.

    Notes
    -----
    This class applies the Bayesian approach to quantifying uncertainty
    in the state of stress at a single point in the subsurface that is
    outlined in [1]_. By "single point" here we mean any volume of the
    subsurface that the user wishes to treat as having a homogeneous
    stress state the. Depending on the application this could be a
    volume with length scale of a log scale (tens of centimeters) or
    in a regional-scale analysis could be a volume with length scale of
    kilometers.

    Currently the vertical stress and pore pressure are taken as
    deterministically known, so that uncertainties in these values
    are not reflected in the uncertainty in the the horizontal
    principal stresses. This may change in a future release, but in
    many applications is not a big limitation because there is
    generally a smaller uncertainty in the vertical stress and pore
    pressure compared to the horizontal principal stresses.

    Currently this class does not contain stress orientation
    information, though this too is planned to change in a future
    release. Though this would only add significant value when methods
    to add different StressState objects are needed and implemented.
    This will be required, for example, to interpolate between
    different points where a StressState object has been parameterized.

    References
    ----------
    [1] Burghardt, J. "Geomechanical Risk Assessment for Subsurface
    Fluid Disposal Operations," Rock Mech Rock Eng 51, 2265–2288 (2018)
    https://doi.org/10.1007/s00603-018-1409-1

    Examples
    --------
    To compute and plot the posterior distribution at a point
    with a frictional faulting constraint, you would do the following:

    >>> from SOSAT import StressState
    >>> from SOSAT.constraints import FaultConstraint
    >>> ss = StressState(1.0,
                         2.5,
                         0.3,
                         depth_unit='km',
                         density_unit='g/cm^3',
                         pressure_unit='MPa')

    >>> sigv = ss.vertical_stress
    >>> fc = FaultConstraint()
    >>> ss.add_constraint(fc)
    >>> fig = ss.plot_posterior()
    >>> plt.savefig("fault_constraint_posterior.png")

    """

    def __init__(self,
                 depth,
                 avg_overburden_density,
                 pore_pressure,
                 depth_unit='m',
                 density_unit='kg/m^3',
                 pressure_unit='MPa',
                 min_stress_ratio=0.4,
                 max_stress_ratio=3.25,
                 nbins=200,
                 stress_unit="MPa"):
        """
        Constructor method
        """
        self._posterior_evaluated = False
        self.stress_unit = stress_unit
        self.depth = depth * units(depth_unit)
        self.depth_unit = depth_unit
        self._avg_overburden_density = \
             avg_overburden_density * units(density_unit)
        self.vertical_stress = (self.depth
                                * self._avg_overburden_density
                                * gravity).to(stress_unit).magnitude
        self.pore_pressure = pore_pressure * units(pressure_unit)

        # convert pore pressure to stress unit in the case that it was
        # passed in with a different unit
        self.pore_pressure = self.pore_pressure.to(stress_unit).magnitude
        self._minimum_stress = min_stress_ratio * self.vertical_stress
        if self._minimum_stress < self.pore_pressure:
            self._minimum_stress = self.pore_pressure
        self._maximum_stress = max_stress_ratio * self.vertical_stress
        self._constraints = []
        # a vector containing the center of each stress bin considered
        sigvec = np.linspace(self._minimum_stress,
                             self._maximum_stress,
                             nbins)

        # create a meshgrid object holding each possible stress state
        shmax_grid, shmin_grid = np.meshgrid(sigvec, sigvec)

        # now create a masked array with the states where the minimum
        # horizontal stress is less than the maximum horizontal stress
        # masked out
        mask = shmin_grid > shmax_grid

        self.shmin_grid = ma.MaskedArray(shmin_grid, mask=mask)
        self.shmax_grid = ma.MaskedArray(shmax_grid, mask=mask)
        # posterior stress distribution, initialized to the
        # uninformative prior where all compressive states are equally
        # likely
        psig = np.ones_like(self.shmin_grid)
        self.psig = ma.MaskedArray(psig, mask=mask)

    def regime(self):
        """
        Computes the scalar regime parameters for each stress state
        included in the class. The scalar regime parameter varies from
        negative one to one. It is defined the vector space where each
        each stress state is represented by a vector whose head lies
        at the coordinate (shmax,shmin) and whose tail lies at the
        spherical stress state where

            shmax = shmin = vertical stress

        The scalar measure of the faulting regime is defined by the
        dot product of the vector for the given stress state and
        the unit vector give by

            shmax = shmin = - 1/sqrt(2)

        Using this definition values of the regime parameter between
        -1 and -sqrt(2)/2 correspond to thrust faulting states,
        values between -sqrt(2)/2 and sqrt(2)/2 correspond to strike-
        slip states, and values between sqrt(2)/2 and +1 correspond
        to normal faulting states.

        Returns
        -------
        2D Numpy MaskedArray with dtype=float
            The array is square with dimensions :attr:`nbins`
        """
        num = 1.0 / np.sqrt(2.0) * (2.0 * self.vertical_stress
                                    - self.shmin_grid
                                    - self.shmax_grid)
        den = ma.sqrt((self.shmax_grid - self.vertical_stress)**2
                       + (self.shmin_grid - self.vertical_stress)**2)
        ret = num / den
        # ensure that the mask has been preserved
        ret.mask = self.shmin_grid.mask
        return ret

    def A_phi_calculate(self):
        """
        Computes the scalar A_phi value for each stress state
        included in the class. The scalar regime parameter varies from
        0 to 1.
        The scalar measure of the faulting regime is defined
        in Simpson (1997):

            phi = (sigma2-sigma3)/ (sigma1-sigma3)
            n = 0, 1, and 2 respectively for normal, strike-slip,
            and reverse faulting regime respectively.
            A_phi = (n+0.5) + (-1)^n*(phi-0.5)

        Using this definition, A_phi value ranges from 0 to 3, where
        [0,1] corresponds to normal faulting regime,
        [1,2] corresponds to strike-slip faulting regime,  and
        [2,3] corresponds to reverse faulting regime.

        Returns
        -------
        2D Numpy MaskedArray with dtype=float
            The array is square with dimensions :attr:`nbins`

        """

        # initiate empty value store
        Sv_grid = self.vertical_stress * np.ones_like(self.shmax_grid)
        A_phi_grid = np.zeros_like(self.shmax_grid)
        sigma1 = np.zeros_like(self.shmax_grid)
        sigma2 = np.zeros_like(self.shmax_grid)
        sigma3 = np.zeros_like(self.shmax_grid)

        # Get three principal stresses
        NFindex = self.vertical_stress > self.shmax_grid
        TFindex = self.shmin_grid > self.vertical_stress
        SSindex = ~NFindex & ~TFindex
        sigma1[NFindex] = Sv_grid[NFindex]
        sigma1[SSindex] = self.shmax_grid[SSindex]
        sigma1[TFindex] = self.shmax_grid[TFindex]
        sigma3[NFindex] = self.shmin_grid[NFindex]
        sigma3[SSindex] = self.shmin_grid[SSindex]
        sigma3[TFindex] = Sv_grid[TFindex]
        sigma2[NFindex] = self.shmax_grid[NFindex]
        sigma2[SSindex] = Sv_grid[SSindex]
        sigma2[TFindex] = self.shmin_grid[TFindex]

        # calculate A_phi
        phi = (sigma2 - sigma3) / (sigma1 - sigma3)
        n_NF = 0
        n_SS = 1
        n_TF = 2
        A_phi_grid[NFindex] = (n_NF + 0.5) + (-1)**n_NF * (phi[NFindex] - 0.5)
        A_phi_grid[SSindex] = (n_SS + 0.5) + (-1)**n_SS * (phi[SSindex] - 0.5)
        A_phi_grid[TFindex] = (n_TF + 0.5) + (-1)**n_TF * (phi[TFindex] - 0.5)

        return A_phi_grid

    def add_constraint(self, constraint):
        """
        Method to add a constraint to the stress state probability
        distribution. Constraints will be applied in the order that
        they are added

        Parameters
        ----------
        constraint : Constraint object
            The object containing the information for the constraint,
            must be a class in the SOSAT.constraints submodule
        """
        self._constraints.append(constraint)
        self._posterior_evaluated = False

    def evaluate_posterior(self):
        """
        Method to evaluate the posterior joint probability density
        given the constraints that have been added. If no constraints
        have been added the prior distribution will be returned.

        Returns
        -------
        2D Numpy MaskedArray with dtype=float
            The array is square with dimensions :attr:`nbins`, where
            each entry in the array contains the probability density
            for the stress state defined by :attr:`shmin_grid` and
            :attr:`shmax_grid`

        Notes
        -----
        The probability density will be properly normalized such that
        the the values of all bins in the distribution will sum to one.
        The prior distribution is homogeneous for all compressive
        stress states considered in the distribution. The stress states
        considered in the distribution are governed by the
        `min_stress_ratio` and `max_stress_ratio` passed into the
        constructor of this class.

        The posterior distribution is computed by recursively applying
        Bayes' law using the likelihood function provided by each
        constraint object that has been added to this class.
        """

        # skip re-evaluation if no new constraints have been added
        # since the last evaluation:
        if not self._posterior_evaluated:

            # initialize with the prior and use log likelihoods to update
            log_posterior = np.log(self.psig)
            for c in self._constraints:
                loglikelihood = c.loglikelihood(self)
                log_posterior = log_posterior + loglikelihood

            self.posterior = np.exp(log_posterior)
            # reset the mask to account for the problem of mask
            # getting changed when ma.log(0)=Masked in the
            # c.loglikelihood(self) function
            # set the posterior to zero where the mask has changed
            indicator = self.posterior.mask != self.shmin_grid.mask
            self.posterior[indicator] = 0
            # now normalize
            tot = ma.sum(self.posterior)
            self.posterior = self.posterior / tot
            self._posterior_evaluated = True

    def plot_posterior(self,
                       figwidth=5.0,
                       figheight=3.5,
                       contour_levels=5,
                       cmap=plt.cm.Greys):
        """
        Makes a contour plot of the joint probability distribution
        of the maximum and minimum horizontal stresses

        Parameters
        ----------
        figwidth : float, optional
            the width of the figure in inches, defaults to 5 inches
        figheight : float, optional
            the height of the figure in inches, defaults to 5 inches
        contour_levels : int, optional
            the number of contour levels desired in the plot
        cmap : colormap object, optional
            a matplotlib colormap object to use to display the
            probability density. Default is matplotlib.pyplot.cm.Greys

        Returns
        -------
        matplotlib.pyplot.Figure object

        Notes
        -----
        The plot is generated with `matplotlib.pyplot.contourf`
        """
        if not self._posterior_evaluated:
            self.evaluate_posterior()

        # Convert data format to bin-centered eval of PDF for plotting
        dsig = self.shmax_grid[0][1] - self.shmax_grid[0][0]
        posterior_PDF = self.posterior / dsig**2.0

        fig = plt.figure(figsize=(figwidth, figheight))
        ax = fig.add_subplot(111)
        im = ax.contourf(self.shmin_grid,
                         self.shmax_grid,
                         posterior_PDF,
                         contour_levels,
                         cmap=plt.cm.Greys)
        plt.colorbar(im, format=ticker.FuncFormatter(fmt))
        ax.set_xlabel("Minimum Horizontal Stress ("
                      + self.stress_unit
                      + ")")
        ax.set_ylabel("Maximum Horizontal Stress ("
                      + self.stress_unit
                      + ")")
        plt.tight_layout()
        return fig

    def get_shmin_marginal(self):
        """
        get the marginal probability distribution for the
        minimum principal stress
        """
        if not self._posterior_evaluated:
            self.evaluate_posterior()

        pshmin = np.sum(self.posterior, axis=1)
        sigvec = np.linspace(self._minimum_stress,
                             self._maximum_stress,
                             np.shape(self.shmin_grid)[0])

        return sigvec, pshmin

    def get_shmax_marginal(self):
        """
        get the marginal probability distribution for the
        maximum principal stress
        """
        if not self._posterior_evaluated:
            self.evaluate_posterior()
        pshmax = np.sum(self.posterior, axis=0)
        sigvec = np.linspace(self._minimum_stress,
                             self._maximum_stress,
                             np.shape(self.shmin_grid)[0])

        return sigvec, pshmax

    def get_shmin_marginal_cdf(self):
        """
        get the marginal cumulative probability function for the
        minimum horizontal stress
        """
        sigvec, pshmin = self.get_shmin_marginal()
        shmin_cdf = np.cumsum(pshmin)
        return sigvec, shmin_cdf

    def get_shmax_marginal_cdf(self):
        """
        get the marginal cumulative probability function for the
        maximum horizontal stress
        """
        sigvec, pshmax = self.get_shmax_marginal()
        shmax_cdf = np.cumsum(pshmax)
        return sigvec, shmax_cdf

    def get_shmin_confidence_intervals(self, confidence):
        """
        return an upper and lower bound within a specified confidence
        interval. Confidence interval should be specified as a fraction
        so that, for example, a 95% confidence interval is expressed
        as confidence=0.95
        """

        sigvec, shmin_cdf = self.get_shmin_marginal_cdf()
        i_low = np.argmax(shmin_cdf > (1.0 - confidence))
        i_high = np.argmax(shmin_cdf > confidence)
        return sigvec[i_low], sigvec[i_high]

    def get_shmax_confidence_intervals(self, confidence):
        """
        return an upper and lower bound within a specified confidence
        interval. Confidence interval should be specified as a fraction
        so that, for example, a 95% confidence interval is expressed
        as confidence=0.95
        """

        sigvec, shmax_cdf = self.get_shmax_marginal_cdf()
        i_low = np.argmax(shmax_cdf > (1.0 - confidence))
        i_high = np.argmax(shmax_cdf > confidence)
        return sigvec[i_low], sigvec[i_high]


class PostOpStressState(StressState):
    """
    A class to contain all data necessary to define the probability
    distribution for all possible stress states at a given point in the
    subsurface at some given post-operation state with a deterministic
    change in pore pressure.
    Attributes
    ----------
    preOp_stress_state : A `SOSAT.StressState` object
        The stress state to use to make the evaluation
    gamma_dist : an object derived from `scipy.stats.rv_continuous`
        A probability distribution for the stress path coefficient
    dP : float
        The change in pore pressure between the pre- and post-operation
        conditions

    Parameters
    ----------
    preOp_stress_state : A `SOSAT.StressState` object
        The stress state to use to make the evaluation
    gamma_dist : an object derived from `scipy.stats.rv_continuous`
        A probability distribution for the stress path coefficient
    dP : float
        The change in pore pressure between the pre- and post-operation
        conditions.
    pressure_unit : str, optional
        Unit of measurement for pressure, see list of units in pint
        package documentation

    Notes
    -----
    This is a subclass of StressState. Objects will have all attributes of
    StressState class, with additional attributes listed above.

    Examples
    --------
    To compute and plot the post-operation posterior distribution at a
    point with a frictional faulting constraint, you would do the
    following:

    >>> from SOSAT import StressState
    >>> from SOSAT import PostOpStressState
    >>> from SOSAT.constraints import FaultConstraint
    >>> from scipy.stats import uniform
    >>> ss = StressState(1.0,
                         2.5,
                         0.3,
                         depth_unit='km',
                         density_unit='g/cm^3',
                         pressure_unit='MPa')

    >>> sigv = ss.vertical_stress
    >>> fc = FaultConstraint()
    >>> ss.add_constraint(fc)
    >>> gamma_dist = uniform(0.4,(0.6-0.4))
    >>> dP = 15.0 # MPa
    >>> postOp_ss = PostOpStressState(ss, gamma_dist, dP)
    >>> fig = postOp_ss.plot_posterior()
    >>> plt.savefig("postOp_posterior.png")

    """
    def __init__(self,
                 preOp_stress_state,
                 gamma_dist,
                 dP,
                 pressure_unit='MPa'):

        """
        Constructor method
        """
        self._posterior_evaluated = False
        self.preOp_stress_state = preOp_stress_state
        self.gamma_dist = gamma_dist
        self.stress_unit = preOp_stress_state.stress_unit
        self.depth = preOp_stress_state.depth
        self.depth_unit = preOp_stress_state.depth_unit
        self._avg_overburden_density = \
            preOp_stress_state._avg_overburden_density
        self._constraints = preOp_stress_state._constraints
        self.pore_pressure = preOp_stress_state.pore_pressure

        # Evaluate the pre-opertaion stress state posterior if not
        # already done
        if not self.preOp_stress_state._posterior_evaluated:
            self.preOp_stress_state.evaluate_posterior()

        # Assuming constant vertical stress, as we have in the risk
        # analysis components of SOSAT
        self.vertical_stress = preOp_stress_state.vertical_stress

        # Convert dP to stress unit in the case that it was passed in
        # with a different unit
        self.dP = dP * units(pressure_unit)
        self.dP = self.dP.to(preOp_stress_state.stress_unit).magnitude

        # Find minimum and maximum post-operation stresses to consider
        # If gamma_dist is uniform, use the min and max values
        # Otherwise, use 3 SD above and below mean
        if isinstance(gamma_dist.dist, type(scipy.stats.uniform)):
            delSigma_small = (self.gamma_dist.mean() - np.sqrt(3.0)
                              * self.gamma_dist.std()) * self.dP
            delSigma_large = (self.gamma_dist.mean() + np.sqrt(3.0)
                              * self.gamma_dist.std()) * self.dP
            if self.dP >= 0.0:
                self._minimum_stress = preOp_stress_state._minimum_stress + \
                                       delSigma_small
                self._maximum_stress = preOp_stress_state._maximum_stress + \
                                       delSigma_large
            else:
                self._minimum_stress = preOp_stress_state._minimum_stress + \
                                       delSigma_large
                self._maximum_stress = preOp_stress_state._maximum_stress + \
                                       delSigma_small
        else:
            delSigma_small = (self.gamma_dist.mean() - 3.0
                              * self.gamma_dist.std()) * self.dP
            delSigma_large = (self.gamma_dist.mean() + 3.0
                              * self.gamma_dist.std()) * self.dP
            if self.dP >= 0.0:
                self._minimum_stress = preOp_stress_state._minimum_stress + \
                                       delSigma_small
                self._maximum_stress = preOp_stress_state._maximum_stress + \
                                       delSigma_large
            else:
                self._minimum_stress = preOp_stress_state._minimum_stress + \
                                       delSigma_large
                self._maximum_stress = preOp_stress_state._maximum_stress + \
                                       delSigma_small

        # A vector containing the center of each stress bin considered
        nbins = np.shape(self.preOp_stress_state.posterior)[0]
        sigvec = np.linspace(self._minimum_stress,
                             self._maximum_stress,
                             nbins)

        # Create a meshgrid object holding each possible stress state
        shmax_grid, shmin_grid = np.meshgrid(sigvec, sigvec)

        # Now create a masked array with the states where the minimum
        # horizontal stress is less than the maximum horizontal stress
        # masked out
        mask = shmin_grid > shmax_grid

        self.shmin_grid = ma.MaskedArray(shmin_grid, mask=mask)
        self.shmax_grid = ma.MaskedArray(shmax_grid, mask=mask)

    def evaluate_posterior(self):
        '''
        The posterior is evaluated using the convolution of the joint
        PDF for pre-operation stress state and the change in stress
        distribution using the stress path coefficient and change in
        pore pressure.
        '''

        # Initialize posterior numpy array
        mask = self.shmin_grid > self.shmax_grid
        self.posterior = np.zeros_like(self.shmax_grid)
        self.posterior = ma.MaskedArray(self.posterior, mask=mask)

        # Find the minimum and maximum gamma value to consider
        if isinstance(self.gamma_dist.dist, type(scipy.stats.uniform)):
            min_gamma = (self.gamma_dist.mean()
                         - np.sqrt(3.0) * self.gamma_dist.std())
            max_gamma = (self.gamma_dist.mean()
                         + np.sqrt(3.0) * self.gamma_dist.std())
        else:
            min_gamma = (self.gamma_dist.mean() - 3.0 * self.gamma_dist.std())
            max_gamma = (self.gamma_dist.mean() + 3.0 * self.gamma_dist.std())

        # Find all the pairs of stress states in the pre-operation PDF
        pre_sig_pairs = list(zip(
                        self.preOp_stress_state.shmax_grid.data.ravel(),
                        self.preOp_stress_state.shmin_grid.data.ravel()))

        # Find associated probabilities in the pre-operation PDF
        # Convert data to PDF format
        preOp_dsig = self.preOp_stress_state.shmax_grid[0][1] \
                   - self.preOp_stress_state.shmax_grid[0][0]
        pre_posterior = list(
                        self.preOp_stress_state.posterior.data.ravel()
                        / preOp_dsig**2)

        # Loop over the gamma dist space
        N_gamma = 50
        d_gamma = (max_gamma - min_gamma) / N_gamma
        gamma_range = [(2 * i + 1) * d_gamma / 2 + min_gamma
                       for i in range(N_gamma)]
        for gamma in gamma_range:

            gamma_prob = self.gamma_dist.pdf(gamma)
            dss_prob = gamma_prob / np.abs(self.dP)

            # Find the required pre-operation stress state
            shmax_preOp = self.shmax_grid - gamma * self.dP
            shmin_preOp = self.shmin_grid - gamma * self.dP
            preOp_pairs = list(zip(shmax_preOp.data.ravel(),
                                   shmin_preOp.data.ravel()))

            # Find the probability of the pre-operation ss
            # Using nearest neighbor interpolation
            index = KDTree(pre_sig_pairs).query(preOp_pairs)
            preOp_prob = [pre_posterior[index[1][i]]
                          for i in range(len(index[1]))]
            preOp_prob = np.reshape(preOp_prob, np.shape(self.shmax_grid))

            # Update preOp_prob to 0 for any point outside of preOp ss space
            preOp_prob[np.array(shmin_preOp
                       < self.preOp_stress_state._minimum_stress)
                       | np.array(shmax_preOp
                       > self.preOp_stress_state._maximum_stress)] \
                       = 0.0

            # Sum the probabilities of all possible paths
            self.posterior += dss_prob * preOp_prob * \
                              (d_gamma * np.abs(self.dP))

        # Convert data in posterior to store total probability in each bin
        # This mimics the data format in the posterior of StressState
        postOp_dsig = self.shmax_grid[0][1] - self.shmax_grid[0][0]
        self.posterior *= postOp_dsig**2

        self._posterior_evaluated = True
