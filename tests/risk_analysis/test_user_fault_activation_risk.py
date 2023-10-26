import pytest
from scipy.stats import uniform
import numpy as np

from SOSAT import StressState
from SOSAT.constraints import FaultConstraint
from SOSAT.risk_analysis import UserPrescribedFaultActivation


def test_upfa():
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

    # Fault Orientation (randomized)
    ss_azi_m = 123.38
    ss_K = 200.0
    strike_m = 333.59
    dip_m = 37.63
    fault_K = 200.0

    upfa = UserPrescribedFaultActivation(ss, dPmax, gamma_dist,
                                         ss_azi_m=ss_azi_m, ss_K=ss_K,
                                         strike_m=strike_m, dip_m=dip_m,
                                         fault_K=fault_K)
    n_shmax, n_fault, n_fault_p, ss_azi, strike_fault, dip_fault =\
        upfa.CreateOrientations()
    pressures, Pfail = upfa.EvaluatePfail()

    print("mean ss_azi = ", np.mean(ss_azi))
    print("mean stike = ", np.mean(strike_fault))
    print("mean dip = ", np.mean(dip_fault))
    print("mean n_shmax[:,0]", np.mean(n_shmax[:, 0]))
    print("mean n_fault_p[:,0]", np.mean(n_fault_p[:, 0]))
    print("pressures= ", pressures)
    print("Pfail= ", Pfail)

    np.mean(ss_azi) == pytest.approx(ss_azi_m, rel=1e-1)
    np.mean(strike_fault) == pytest.approx(strike_m, rel=1e-1)
    np.mean(dip_fault) == pytest.approx(dip_m, rel=1e-1)
    np.mean(n_shmax[:, 0]) == pytest.approx(-0.548, rel=1e-1)
    np.mean(n_fault_p[:, 0]) == pytest.approx(0.305, rel=1e-1)
    pressures[0] == pytest.approx(12.227, rel=1e-2)
    Pfail[0] == pytest.approx(0.0533, rel=0.5e-1)
    pressures[-1] == pytest.approx(ss.pore_pressure + dPmax)
    Pfail[-1] == pytest.approx(0.5153, rel=0.5e-1)

    fig = upfa.PlotFailureProbability()

    fig.savefig("User_Fault_Activation_Probability.png")
