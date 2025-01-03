import pytest
from SOSAT import StressState
from SOSAT import PostOpStressState
from SOSAT.constraints import FaultConstraint
from scipy.stats import uniform

def test_PostOpStressState():

	# depth in meters
	depth = 1228.3
	# density in kg/m^3
	avg_overburden_density = 2580.0
	# pore pressure gradient in MPa/km
	pore_pressure_grad = 9.955

	pore_pressure = pore_pressure_grad * (1.0 / 1000) * depth

	ss = StressState(depth=depth,
	                avg_overburden_density=avg_overburden_density,
	                pore_pressure=pore_pressure, 
	                nbins = 50)

	fc = FaultConstraint()
	ss.add_constraint(fc)

	gamma_dist = uniform(0.4,(0.6-0.4))
	dP = 15.0 # MPa
	postOp_ss = PostOpStressState(ss, gamma_dist, dP)

	postOp_ss.evaluate_posterior()

	sig = postOp_ss.posterior[20,20]
	sig == pytest.approx(0.001829)
