import numpy as np
from numpy import ma
import pint

from .constraint_base import StressConstraint

units = pint.UnitRegistry()


def MCfail(sig1, sig3, phi, Co):
    """
    Mohr Coulomb failure criterion

    Parameters
    ----------
    sig1 : float, array_like
        maximum principal stress
    sig3 : float, array_like
        minimum principal stress
    phi : float
        friction angle
    Co : float
        cohesion

    Returns
    -------
    f : float or array_like
        The value of the failure function, which is defined so that
        negative values indicate no failure and positive values
        indicate failure, a zero value is at incipient failure
    """
    tau = (sig1 - sig3) / 2
    sig_m = (sig1 + sig3) / 2

    return tau - sig_m * np.sin(phi) - Co * np.cos(phi)


class BreakoutConstraint(StressConstraint):
    """
    A class used to constrain the stress state by the existence or non
    existence of borehole breakouts at the location being analyzed. If
    breakouts exists that generally indicates that the maximum
    horizontal stress is much larger than the minimum horizontal
    stress, if not breakout exists that puts a limit on how large the
    maximum horizontal stress could be; the magnitude of that limit
    depends on the strength of the rock, drilling mud weight,
    temperature, etc. Significant uncertainty in these parameters,
    especially the rock strength and mud weight makes the constraint
    provided by a breakout analysis much weaker. This class seeks to
    account for these uncertainties in the posterior stress
    distribution

    Attributes
    ----------
    No public attributes

    Parameters
    ----------
    breakout_exists : bool
        Indication whether or not breakouts exist
    UCS_dist : subclass of `scipy.stats.rv_continuous`
        A probability distribution for the unconfined compressive
        strength. Shoudl use the same pressure unit as is used by
        the mud pressure and Young's modulus, but this can be any
        unit as specified though the optional `pressure_unit`
        parameter; this shoudl be an estimate for the *minimum*
        UCS in the zone of interest since that is what would govern
        the formation of breakouts, not the average or representative
        value
    rock_friction_angle_dist : subclass of `scipy.stats.rv_continuous`
        A probabiilty distribution for the rock friction angle.
    rock_friction_angle_units : str
        The unit used for the rock friction angle. Should be an angular
        unit recognized by `pint.UnitRegistry` (i.e. 'deg' for degrees,
        'radians' for radians
    mud_pressure_dist : subclass of `scipy.stats.rv_continuous`
        The probability distribution for the minimum mud pressure
        experienced by the relevant section of borehole from the time
        that the well was drilled until the log used to identify the
        presence or absence of breakouts was run; mud pressure should
        be specified in the same pressure unit as is used for UCS and
        Young's modulus, but this can be any unit as specified though
        the optional `pressure_unit` parameter, which defaults to 'Pa';
        conversion from mud weight must be performed by the user of
        this class
    mud_temperature_dist :  subclass of `scipy.stats.rv_continuous`
        The probability distribution for the maximum mud temperature;
        the maximum value is of interest rather than the average value
        since the formation of a breakout is governed by the maximum
        value only
    formation_temperature : float
        Formation temperature, which is taken as deterministic since
        it is usually not highly uncertain
    YM : float
        Formation Young's Modulus, which is taken as deterministic
        since the formation of breakouts is only weakly dependent
        on this parameter; should be specified in the same pressure
        unit as is used for mud pressure and Young's modulus, but
        this can be any unit as specified though the optional
        `pressure_unit` parameter, which defaults to 'Pa'
    PR : float
        Formation Poisson's Ratio, which is taken as deterministic
        since the formation of breakouts is only weakly dependent
        on this parameter
    CTE : float
        Formation coefficient of thermal expansion, which is taken
        as deterministic since the formation of breakouts is only
        weakly dependent on this parameter
    pressure_unit : str, optional
        The unit used for UCS and Young's modulus; should be a unit
        recognized by `pint.UnitRegistry`; defaults to 'Pa'

    Notes
    -----
    This class uses a Mohr Coulomb (MC) failure criterion to evaluate
    the probability of a breakout occuring. The Mohr Coulomb criterion
    uses two parameters, most often a friction angle and cohesion. In
    this class the unconfined compressive strength (UCS) and friction
    angle were chosen instead of the cohesion and friction angle. This
    is because the UCS is a more commonly known and intuitive property,
    and the cohesion MC parameter can be calculated given the UCS and
    friction angle.

    While this class allows users to use any probability distribution
    that derives from the `scipy.stats.rv_continuous` class, users are
    cautioned against using any distribution that has finite
    probability density for negative parameter values, since negative
    strength values are not physically meaningful. Therefore, lognormal
    distributions are more appropriate than a normal distribution, for
    example.
    """

    def __init__(self,
                 breakout_exists,
                 UCS_dist,
                 rock_friction_angle_dist,
                 rock_friction_angle_units,
                 mud_pressure_dist,
                 mud_temperature_dist,
                 formation_temperature,
                 YM,
                 PR,
                 CTE,
                 pressure_unit='Pa'):
        """
        Constructor method
        """

        self._breakout_exists = breakout_exists
        self._UCS_dist = UCS_dist
        self._rock_friction_angle_dist = rock_friction_angle_dist
        self._rock_friction_angle_units = rock_friction_angle_units
        self._mud_pressure_dist = mud_pressure_dist
        self._mud_temperature_dist = mud_temperature_dist
        self._formation_temperature = formation_temperature
        self._YM = YM * units(pressure_unit)
        self._PR = PR
        self._CTE = CTE
        self._pressure_unit = pressure_unit

    def loglikelihood(self, ss):
        """
        Computes the loglikelihood of each stress state given the
        presence or absence of breakouts, formation properties, and mud
        properties specified

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

        # nominal maximum hoop stress using Kirch solution
        # which will have units of ss.stress_unit

        sig1_nominal = 3.0 * ss.shmax_grid - ss.shmin_grid \
                       - 2.0 * ss.pore_pressure
        # compute the thermoelastic factor, which will have
        # units of self._pressure_unit/(delta self.temperature unit)
        TEF = (self._CTE * self._YM / (1.0 - self._PR))

        # since all temperature-based quantities in the class are
        # assumed to be consistent, we do not include pint temperature
        # units explicitly the way we do for pressure/stress. This means
        # that TEF will only have pressure units. We convert it to
        # ss.stress_units here to avoid repeated conversions inside the
        # Monte Carlo loop

        TEF = TEF.to(ss.stress_unit).magnitude

        # use a Monte Carlo sampling scheme to evaluate the probability
        # of a breakout forming

        # NBO is a masked array that is used to keep a tally of the
        # number of breakouts for each stress state during the Monte
        # Carlo sampling
        NBO = ma.zeros(np.shape(ss.shmin_grid), dtype=np.int32)
        # PBO_new is the updated estimate for the probability of
        # a breakout forming at each stress state after the most recent
        # realizations
        PBO_new = ma.zeros(np.shape(ss.shmin_grid), dtype=np.float64)
        Ntotal = 0
        converged = False
        iter = 0
        while not converged:
            # perform 500 iterations at a time and then see if the probability
            # has changes meaningfully
            for i in range(0, 500):
                # draw random samples
                mud_pressure_i = self._mud_pressure_dist.rvs() \
                    * units(self._pressure_unit)
                # convert to the stress unit of ss
                mud_pressure_i = mud_pressure_i \
                                 .to(ss.stress_unit).magnitude
                # no unit conversion is needed since all members of
                # this class should have consistent temperature units
                mud_temperature_i = self._mud_temperature_dist.rvs()

                UCS_i = self._UCS_dist.rvs() * units(self._pressure_unit)
                # convert to the stress unit of ss
                UCS_i = UCS_i.to(ss.stress_unit).magnitude
                # friction parameter in the specified units
                rock_friction_i = self._rock_friction_angle_dist.rvs() \
                                  * units(self._rock_friction_angle_units)
                # convert to radians to be used in the numpy functions
                rock_friction_i = rock_friction_i.to('radians').magnitude

                # convert into more convenient parameters
                # phi_i = np.arctan(rock_friction_i)
                # Co_i = UCS_i * 0.5 * (1.0 - np.sin(phi_i)) / np.cos(phi_i)
                Co_i = UCS_i * 0.5 * (1.0 - np.sin(rock_friction_i)) \
                       / np.cos(rock_friction_i)

                # these should both have the same units now and the pint
                # unit should have been stripped
                deltaP = mud_pressure_i - ss.pore_pressure

                sig3 = deltaP

                # this will have units of delta temperature which is
                # assumed to be consistent between the user-supplied
                # mud temperature and formation temperature
                deltaT = self._formation_temperature - mud_temperature_i

                # compute the mud pressure effect on the nominal maximum
                # hoop stress
                sig1 = sig1_nominal - deltaP - TEF * deltaT

                # the stresses going in here shoudl all have had the pint
                # units stripped after ensuring they were compatible
                BO = MCfail(sig1, sig3, rock_friction_i, Co_i)
                # increment the breakout count at locations where BO > 0.0
                NBO[BO > 0.0] += 1
            # increment the iteration count and realization count
            iter += 1
            Ntotal += 500

            # if there have been more than two iterations we can
            # evalute the change in the probability of breakout
            # formation by comparins how much the last 500 iterations
            # have changed the probability
            if iter > 2:
                PBO_old = PBO_new
                PBO_new = NBO / Ntotal
                err = ma.MaskedArray.max(PBO_new - PBO_old)
                # if the difference between the two is less than 1%
                # then the ieration has converged
                if err < 0.01:
                    converged = True
                    print("breakout Monte Carlo iteration converged after ",
                          iter,
                          " iterations")
        # return the most updated estimate for the likelihood of
        # breakout formation at each stress state
        if self._breakout_exists:
            with np.errstate(divide='ignore'):
                loglikelihood = np.log(PBO_new)
            return loglikelihood
        else:
            with np.errstate(divide='ignore'):
                loglikelihood = np.log1p(- PBO_new)
            return loglikelihood
