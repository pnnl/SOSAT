import pytest
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from SOSAT import StressState
from SOSAT.constraints import StressMeasurement
from SOSAT.constraints import FaultConstraint
from SOSAT.samplers import RejectionSampler


def fmt(x, pos):
    """
    A utility function to improve the formatting of
    plot labels
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


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

# generate samples
sampler = RejectionSampler(ss)
Nsamples = int(1e6)
shmin, shmax, sv = sampler.GenerateSamples(Nsamples)

assert len(shmin) == Nsamples
figwidth = 4.0
fig2 = plt.figure(figsize=(figwidth, figwidth * 0.7))

smin = ss.shmin_grid[0, 0]
smax = ss.shmax_grid[0, -1]
plt.hist2d(shmin,
           shmax,
           density=True,
           range=[[smin, smax], [smin, smax]],
           bins=(200, 200),
           cmap=plt.cm.Greys)

plt.colorbar(format=ticker.FuncFormatter(fmt))
plt.xlabel("Minimum Horizontal Stress (MPa)")

plt.ylabel("Maximum Horizontal Stress (MPa)")

plt.tight_layout()
plt.savefig("faulting_stress_measurement_constraint_posterior_samples.png")
