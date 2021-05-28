import pytest
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
import pint

from SOSAT import StressState
from SOSAT.constraints import StressMeasurement, FaultConstraint

units = pint.UnitRegistry()

# depth in feet
depth = 8520.0 * units('ft')
# density in kg/m^3
avg_overburden_density = 2601.27615 * units('kg/m^3')
# pore pressure gradient in MPa/km
pore_pressure_grad = 9.72686 * units('MPa/km')

pore_pressure = pore_pressure_grad * depth

ss = StressState(depth=depth.to('ft').magnitude,
                 avg_overburden_density=avg_overburden_density
                               .to('lb/ft^3').magnitude,
                 pore_pressure=pore_pressure.to('psi').magnitude,
                 depth_unit='ft',
                 density_unit='lb/ft^3',
                 pressure_unit='psi',
                 stress_unit='psi')


meas_shmin_ul = (0.62 * units('psi/ft')
                        * depth) \
                        .to('psi').magnitude
meas_shmin_ll = meas_shmin_ul - 500.0
meas_shmin_dist = uniform(meas_shmin_ll,
                          scale=(meas_shmin_ul
                                 - meas_shmin_ll))
ss.add_constraint(
         StressMeasurement(meas_shmin_dist))

# ss.add_constraint(FaultConstraint())
fig = ss.plot_posterior()
plt.savefig("stress_measurement_constraint_posterior.png")

shmin_ll, shmin_ul = ss.get_shmin_confidence_intervals(0.99)
print("smin_ll= ", shmin_ll)
print("smin_ul= ", shmin_ul)
