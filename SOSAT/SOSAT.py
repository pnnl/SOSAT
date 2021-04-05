import pint

units = pint.UnitRegistry()
"""Main module."""

gravity = 9.81 * units('m/s^2')


class StressState:
    """A class to contain all data necessary to define the probability
    distribution for all possible stress states at a given point in the
    subsurface.

    :param depth: the true vertical depth of the point being analyzed
    :type  depth: float
    :param average_overburden_density: average mass density of all overlying
         formations
    :type average_overburden_density: float
    :param depth_unit: unit of measurement for depth
    :type depth_unit: str, optional, see list of units in pint package
         documentation
    :param density_unit: unit of measurement for mass density
    :type density_unit: str, optional, see list of units in pint package
         documentation
    """

    def __init__(self,
                 depth,
                 avg_overburden_density,
                 depth_unit='m',
                 density_unit='kg/m^3'):
        """Constructor method
        """

        self.depth = depth * units(depth_unit)
        self.avg_overburden_density = \
             avg_overburden_density * units(density_unit)
        self.vertical_stress = self.depth \
                             * self.avg_overburden_density \
                             * gravity
        print("vertical_stress= ", self.vertical_stress.to('MPa'))
