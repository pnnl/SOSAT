from .core import np, ma, units, gravity


class StressState:
    """A class to contain all data necessary to define the probability
    distribution for all possible stress states at a given point in the
    subsurface.

    :param depth: the true vertical depth of the point being analyzed
    :type  depth: float
    :param average_overburden_density: average mass density of all overlying
        formations
    :type average_overburden_density: float
    :param pore_pressure: formation pore pressure
    :type pore_pressure: float
    :param depth_unit: unit of measurement for depth
    :type depth_unit: str, optional, see list of units in pint package
        documentation
    :param density_unit: unit of measurement for mass density
    :type density_unit: str, optional, see list of units in pint package
        documentation
    :param pressure_unit: unit of measurement for pressure
    :type pressure_unit: str, optional, see list of units in pint package
        documentation
    :param min_stress_ratio: minimum stress included in the analysis expressed
        as a fraction of the vertical stress. Default value is 0.4
    :type min_stress_ratio: float
    :param max_stress_ratio: maximum stress included in the analysis expressed
        as a fraction of the vertical stress. Default value is 2.0
    :type max_stress_ratio: float
    :param nbins: number of bins to use for the horizontal principal
        stresses. The same number is used for both horizontal
        principal stresses. Optional with a default value of 200
    :type nbins: integer
    """

    def __init__(self,
                 depth,
                 avg_overburden_density,
                 pore_pressure,
                 depth_unit='m',
                 density_unit='kg/m^3',
                 pressure_unit='MPa',
                 min_stress_ratio=0.4,
                 max_stress_ratio=2.0,
                 nbins=200,
                 stress_unit="MPa"):
        """Constructor method
        """

        self.depth = depth * units(depth_unit)
        self.avg_overburden_density = \
             avg_overburden_density * units(density_unit)
        self.vertical_stress = (self.depth
                                * self.avg_overburden_density
                                * gravity).to(stress_unit).magnitude
        self.pore_pressure = pore_pressure * units(pressure_unit)
        self.pore_pressure = self.pore_pressure.to(pressure_unit).magnitude
        self.minimum_stress = min_stress_ratio * self.vertical_stress
        self.maximum_stress = max_stress_ratio * self.vertical_stress

        # a vector containing the center of each stress bin considered
        sigvec = np.linspace(self.minimum_stress,
                             self.maximum_stress,
                             nbins)

        # create a meshgrid object holding each possible stress state
        shmax_grid, shmin_grid = np.meshgrid(sigvec, sigvec)

        # now create a masked array with the states where the minimum
        # horizontal stress is less than the maximum horizontal stress
        # masked out
        mask = shmin_grid > shmax_grid

        self.shmin_grid = ma.MaskedArray(shmin_grid, mask=mask)
        self.shmax_grid = ma.MaskedArray(shmax_grid, mask=mask)

        print("vertical_stress= ", self.vertical_stress, stress_unit)

    def regime(self):
        """Computes the scalar regime parameters for each stress state
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

        :return: An array containing the scalar faulting regime for
            each stress state included in the StressState object.

        :rtype: array of the same shape as the stress arrays contained
            in the StressState class
        """
        num = 1.0 / np.sqrt(2.0) * (2.0 * self.vertical_stress
                                    - self.shmin_grid
                                    - self.shmax_grid)
        den = ma.sqrt((self.shmax_grid - self.vertical_stress)**2
                       + (self.shmin_grid - self.vertical_stress)**2)
        arg = num / den
        return arg
