from logging import log
import numpy as np
from numpy import ma
import pint
from .constraint_base import StressConstraint


units = pint.UnitRegistry()


class DITFConstraint(StressConstraint):
    """
    A class used to constrain the stress state by the existence or non
    existence of drilling-induced tensile fractures (DITF) at the
    location being analyzed. Depending on the mud and formation
    temperatures, mud weights, and rock strength and whether or
    not significant mud losses were observed, if DITF's exist it
    generally indicates that the maximum horizontal stress is much
    larger than the minimum principal stress.

    Attributes
    ----------
    No public attributes

    Parameters
    ----------
    DITF_exists : bool
        Indication whether or not DITF exist
    mud_pressure_dist : subclass of `scipy.stats.rv_continuous`
        The probability distribution for the maximum mud pressure
        experienced by the relevant section of borehole from the time
        that the well was drilled until the log used to identify the
        presence or absence of breakouts was run; mud pressure should
        be specified in the same pressure unit as is used for UCS and
        Young's modulus, but this can be any unit as specified though
        the optional `pressure_unit` parameter, which defaults to 'Pa';
        conversion from mud weight must be performed by the user of
        this class
    mud_temperature_dist :  subclass of `scipy.stats.rv_continuous`
        The probability distribution for the minimum mud temperature;
        the minimum value is of interest rather than the average value
        since the formation of a DITF is governed by the minimum
        value only
    tensile_strength_dist : subclass of `scipy.stats.rv_continuous`
        The probability distribution for minimum the minimum tensile
        strength in the zone being analyzed. DITFs will form at the
        weakest portion of the well for a given stress state, so
        whether they form or not is dependent on the minimum tensile
        strength rather than an average representative value
    formation_temperature : float
        Formation temperature, which is taken as deterministic since
        it is usually not highly uncertain
    YM : float
        Formation Young's Modulus, which is taken as deterministic
        since the formation of DITF is only weakly dependent
        on this parameter; should be specified in the same pressure
        unit as is used for mud pressure and Young's modulus, but
        this can be any unit as specified though the optional
        `pressure_unit` parameter, which defaults to 'Pa'
    PR : float
        Formation Poisson's Ratio, which is taken as deterministic
        since the formation of DITFs is only weakly dependent
        on this parameter
    CTE : float
        Formation coefficient of thermal expansion, which is taken
        as deterministic since the formation of DITF is only
        weakly dependent on this parameter
    pressure_unit : str, optional
        The unit used for UCS and Young's modulus; should be a unit
        recognized by `pint.UnitRegistry`; defaults to 'Pa'

    Notes
    -----
    While this class allows users to use any probability distribution
    that derives from the `scipy.stats.rv_continuous` class for the mud
    temperature, pressure, and formation tensile strength, users are
    cautioned against using any distribution that has finite
    probability density for negative parameter values, since negative
    strength values are not physically meaningful. Therefore, lognormal
    distributions are more appropriate than a normal distribution, for
    example.
    """

    def __init__(self,
                 DITF_exists,
                 mud_pressure_dist,
                 mud_temperature_dist,
                 tensile_strength_dist,
                 formation_temperature,
                 YM,
                 PR,
                 CTE,
                 pressure_unit='Pa'):
        """
        Constructor method
        """
        self._DITF_exists = DITF_exists
        self._mud_pressure_dist = mud_pressure_dist
        self._mud_temperature_dist = mud_temperature_dist
        self._tensile_strength_dist = tensile_strength_dist
        self._formation_temperature = formation_temperature
        self._YM = YM * units(pressure_unit)
        self._PR = PR
        self._CTE = CTE
        self._pressure_unit = pressure_unit

    def loglikelihood(self, ss):
        """
        Computes the likelihood of each stress state given the presence
        or absence of DITFs, formation and mud properties specified.

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
        # compute stress with balanced mud and no temperature difference
        sig_nominal = 3.0 * ss.shmin_grid - ss.shmax_grid \
                      - 2.0 * ss.pore_pressure

        # compute thermoelastic factor
        TEF = self._CTE * self._YM / (1.0 - self._PR)

        # since all temperature-based quantities in the class are
        # assumed to be consistent, we do not include pint temperature
        # units explicitly the way we do for pressure/stress. This means
        # that TEF will only have pressure units. We convert it to
        # ss.stress_units here to avoid repeated conversions inside the
        # Monte Carlo loop

        TEF = TEF.to(ss.stress_unit).magnitude

        # use a Monte Carlo sampling scheme to evaluate the probability
        # of a DITF forming

        NDITF = ma.zeros(np.shape(ss.shmin_grid), dtype=np.int32)

        PDITF_new = ma.zeros(np.shape(ss.shmin_grid), dtype=np.float64)
        Ntotal = 0
        converged = False
        iter = 0
        while not converged:
            # perform 500 iterations at a time and then see if the
            # probabiliity has changed meaningfully
            for i in range(0, 500):
                mud_pressure_i = self._mud_pressure_dist.rvs() \
                                 * units(self._pressure_unit)
                # convert to the stress unit of ss
                mud_pressure_i = mud_pressure_i \
                                 .to(ss.stress_unit).magnitude
                # no unit conversion is needed since all members of
                # this calss should have consistent temperature units
                mud_temperature_i = self._mud_temperature_dist.rvs()

                TS_i = self._tensile_strength_dist.rvs() \
                       * units(self._pressure_unit)
                # convert to stress unit of ss
                TS_i = TS_i.to(ss.stress_unit).magnitude

                deltaP = mud_pressure_i - ss.pore_pressure
                deltaT = self._formation_temperature - mud_temperature_i
                DITF = sig_nominal - deltaP - TEF * deltaT + TS_i
                NDITF[DITF < 0.0] += 1
            iter += 1
            Ntotal += 500
            if iter > 2:
                PDITF_old = PDITF_new
                PDITF_new = NDITF / Ntotal
                err = ma.MaskedArray.max(PDITF_new - PDITF_old)
                if err < 0.01:
                    converged = True
                    print("DITF Monte Carlo iteration converged after ",
                          iter,
                          " iterations")
        # return the most updated estimate for the likelihood of
        # DITF formation at each stress state
        if self._DITF_exists:
            with np.errstate(divide='ignore'):
                loglikelihood = np.log(PDITF_new)
            return loglikelihood
        else:
            # we should change this to do the calculation using
            # log probabilities and np.log1p to improve numerical
            # precision when PDITF_new is close to 1.0
            with np.errstate(divide='ignore'):
                loglikelihood = np.log1p(- PDITF_new)
            return loglikelihood
