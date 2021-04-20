import pytest
from scipy.stats import lognorm
import numpy as np
import matplotlib.pyplot as plt

from SOSAT import StressState
from SOSAT.constraints import FaultingRegimeConstraint
from SOSAT.constraints import SU

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


frc = FaultingRegimeConstraint(SU(w_NF=100.0, w_SS=50.0, w_TF=5.0,
                                  theta1=np.sqrt(2.0) * 0.5, k1=300.0,
                                  theta2=-np.sqrt(2.0) * 0.5, k2=300.0))

ss.add_constraint(frc)

fig = ss.plot_posterior()
plt.savefig("fault_regime_constraint_posterior.png")
