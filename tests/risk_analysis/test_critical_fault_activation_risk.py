import pytest
from scipy.stats import uniform
import numpy as np

from SOSAT import StressState
from SOSAT.constraints import StressMeasurement
from SOSAT.constraints import FaultConstraint
from SOSAT.risk_analysis import CriticalFaultActivation


def test_cfa():
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
    dPmax = 4.0
    gamma_dist = uniform(0.4, (0.6 - 0.4))
    cfa = CriticalFaultActivation(ss, dPmax, gamma_dist)
    shmin, shmax, sv = cfa.SampleStressPoints()
    pressures, Pfail = cfa.EvaluatePfail(
        shmin=shmin, shmax=shmax, sv=sv)
    print("pressures= ", pressures)
    print("Pfail= ", Pfail)
    pressures[0] == pytest.approx(12.227)
    Pfail[0] == pytest.approx(0.1316)

    pressures[-1] == pytest.approx(ss.pore_pressure + dPmax)
    Pfail[-1] == pytest.approx(0.2043)

    fig = cfa.PlotFailureProbability()

    fig.savefig("Critical_Fault_Activation_Probability.png")
