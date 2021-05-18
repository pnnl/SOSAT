import pytest
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt

from SOSAT import StressState
from SOSAT.constraints import StressMeasurement
from SOSAT.constraints import FaultConstraint

# depth in meters
depth = 1228.3
# density in kg/m^3
avg_overburden_density = 2580.0
# pore pressure gradient in MPa/km
pore_pressure_grad = 9.955

pore_pressure = pore_pressure_grad * (1.0 / 1000) * depth

ss = StressState(depth=depth,
                 avg_overburden_density=avg_overburden_density,
                 pore_pressure=pore_pressure)
fc = FaultConstraint()
ss.add_constraint(fc)

smc = StressMeasurement(shmin_dist=uniform(loc=25.0,
                                           scale=5.0))

ss.add_constraint(smc)

fig = ss.plot_posterior()
plt.savefig("faulting_stress_measurement_constraint_posterior.png")
