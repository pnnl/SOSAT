import pytest
from scipy.stats import lognorm, uniform
import matplotlib.pyplot as plt
import pint
from SOSAT import StressState
from SOSAT.constraints import FaultConstraint

units = pint.UnitRegistry()

# depth in feet
depth = 7961 * units('ft')

# vertical stress calculated from integration of density log
sigV = 1.131479994 * units('psi/ft') * depth

g = 9.81 * units('m/s^2')

# equivalent average density
# density in kg/m^3
avg_overburden_density = (sigV / (g * depth)).to('kg/m^3')

pore_pressure_grad = 0.466 * units('psi/ft')
# pore pressure gradient in MPa/km


pore_pressure = (pore_pressure_grad * depth).to('psi')

ss = StressState(depth=depth.to('ft').magnitude,
                 avg_overburden_density=avg_overburden_density
                                        .to('lb/ft^3').magnitude,
                 pore_pressure=pore_pressure
                               .to('psi').magnitude,
                 nbins=800,
                 depth_unit='ft',
                 density_unit='lb/ft^3',
                 pressure_unit='psi',
                 stress_unit='psi')

# friction_mu = 0.7
# friction_std = 0.15
# mudist = lognorm(scale=friction_mu,
#                  s=friction_std)
# fc = FaultConstraint(mudist)
fc = FaultConstraint()

ss.add_constraint(fc)

fig = ss.plot_posterior()
plt.savefig("fault_constraint_posterior2.png", dpi=300)

fig = plt.figure()
ax = fig.add_subplot(111)

shmin, pshmin = ss.get_shmin_marginal()
ax.plot(shmin, pshmin, "k")
plt.savefig("fault_constraint_shmin_marginal_posterior2.png", dpi=300)

shmin_ll, shmin_ul = ss.get_shmin_confidence_intervals(0.9999)
print("shmin_ll= ", shmin_ll)
print("shmin_ll= ", (shmin_ll * units('MPa')).to('psi'))
print("ss.pore_pressure= ", ss.pore_pressure)
print("ss.vertical_stress= ", ss.vertical_stress)
assert ss.vertical_stress == pytest.approx(9007.7, abs=1.0e-1)
assert ss.pore_pressure == pytest.approx(3709.8, abs=1.0e-1)
assert shmin_ll == pytest.approx(4733.7, abs=1.0e-1)
