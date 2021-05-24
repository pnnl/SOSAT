import pytest
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

from SOSAT import StressState
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
# friction_mu = 0.7
# friction_std = 0.15
# mudist = lognorm(scale=friction_mu,
#                 s=friction_std)
# fc = FaultConstraint(mudist)
fc = FaultConstraint()

ss.add_constraint(fc)

fig = plt.figure()
ax = fig.add_subplot(111)

shmin, pshmin = ss.get_shmin_marginal()
ax.plot(shmin, pshmin, "k")
plt.savefig("fault_constraint_shmin_marginal_posterior.png")

# test that the marginal sums to 1

psum = np.sum(pshmin)

assert psum == pytest.approx(1.0)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

shmin, pshmax = ss.get_shmax_marginal()
ax2.plot(shmin, pshmax, "k")
plt.savefig("fault_constraint_shmax_marginal_posterior.png")

# test that the marginal sums to 1

psum = np.sum(pshmax)

assert psum == pytest.approx(1.0)
