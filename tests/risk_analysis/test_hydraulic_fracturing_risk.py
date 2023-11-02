import pytest
from scipy.stats import uniform
import numpy as np

from SOSAT import StressState
from SOSAT.constraints import FaultConstraint
from SOSAT.risk_analysis import HydraulicFracturing


def test_hf():
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

    dPmax = 10.0
    gamma_dist = uniform(0.4, (0.6 - 0.4))

    # Tensile strength distribution
    T_dist = uniform(0.0, 5.0)

    hf = HydraulicFracturing(ss, dPmax, gamma_dist)

    pressures, Pfail = hf.EvaluatePfail()

    print("pressures= ", pressures)
    print("Pfail= ", Pfail)

    pressures[0] == pytest.approx(12.227, rel=1e-2)
    Pfail[0] == pytest.approx(0.0, rel=1e-2)
    pressures[-1] == pytest.approx(ss.pore_pressure + dPmax)
    Pfail[-1] == pytest.approx(3.212e-03, rel=1e-2)

    fig = hf.PlotFailureProbability()

    fig.savefig("HF_Probability.png")

    hf_T = HydraulicFracturing(ss, dPmax, gamma_dist, T_dist)

    pressures_T, Pfail_T = hf_T.EvaluatePfail()

    print("pressures= ", pressures_T)
    print("Pfail= ", Pfail_T)

    pressures_T[0] == pytest.approx(12.227, rel=1e-2)
    Pfail_T[0] == pytest.approx(0.0, rel=1e-2)
    pressures_T[-1] == pytest.approx(ss.pore_pressure + dPmax)
    Pfail_T[-1] == pytest.approx(4.00e-04, rel=1e-2)

    fig = hf_T.PlotFailureProbability()

    fig.savefig("HF_Probability_with_Tesnile_Strength.png")
