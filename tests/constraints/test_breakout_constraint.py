import pytest
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

from SOSAT import StressState
from SOSAT.constraints import BreakoutConstraint

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


# the cohesion is chosen to be uniform between 25 and 35 MPa
# and a friction angle of about 10 degrees
phi = np.deg2rad(10.0)
C_ll = 25.0
C_ul = 35.0
UCS_ll = C_ll * np.cos(phi) / (1.0 - np.sin(phi))
UCS_ul = C_ul * np.cos(phi) / (1.0 - np.sin(phi))
UCS_dist = uniform(loc=UCS_ll, scale=(UCS_ul - UCS_ll))

# this makes the friction angle deterministic, by essentiall
# making it a Dirac Delta pdf
friction_angle_dist = uniform(loc=phi, scale=0.0)

mud_pressure_dist = uniform(loc=16.32, scale=(17.32 - 16.32))
mud_temperature_dist = uniform(loc=35.0, scale=(50.0 - 35.0))
formation_temperature = 30.7
# YM in MPa, so 19,000 MPa = 19 GPa
YM = 19.0e3
PR = 0.25
CTE = 2.4e-6
bc = BreakoutConstraint(breakout_exists=False,
                        UCS_dist=UCS_dist,
                        rock_friction_angle_dist=friction_angle_dist,
                        rock_friction_angle_units='radians',
                        mud_pressure_dist=mud_pressure_dist,
                        mud_temperature_dist=mud_temperature_dist,
                        formation_temperature=formation_temperature,
                        YM=YM,
                        PR=PR,
                        CTE=CTE,
                        pressure_unit='MPa')
ss.add_constraint(bc)
fig = ss.plot_posterior()
plt.savefig("breakout_constraint_posterior.png")
