import pytest
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

from SOSAT import StressState
from SOSAT.constraints import FaultConstraint


def test_confidence_interval():
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

    fig = plt.figure()
    ax = fig.add_subplot(111)
    shmin, pshmin = ss.get_shmin_marginal()
    ax.plot(shmin, pshmin, "k")
    plt.savefig("fault_constraint_shmin_marginal_posterior.png")

    # test that for a confidence interval < 50% the lower bound
    # is less than the upper bound and that it is also greater
    # than a 95% confidence interval
    shmin_ll_95, shmin_ul_95 = ss.get_shmin_confidence_intervals(0.95)
    shmin_ll_20, shmin_ul_20 = ss.get_shmin_confidence_intervals(0.20)
    print("shmin_ll_95= ", shmin_ll_95)
    print("shmin_ll_20= ", shmin_ll_20)
    assert shmin_ll_95 == pytest.approx(21.78505, abs=1.0e-5)
    print("shmin_ul_95= ", shmin_ul_95)
    print("shmin_ul_20= ", shmin_ul_20)
    assert shmin_ul_95 == pytest.approx(73.87703, abs=1.0e-5)
    assert shmin_ll_20 < shmin_ul_20
    assert shmin_ll_20 > shmin_ll_95

    shmax_ll_95, shmax_ul_95 = ss.get_shmax_confidence_intervals(0.95)
    shmax_ll_20, shmax_ul_20 = ss.get_shmax_confidence_intervals(0.20)
    assert shmax_ll_95 == pytest.approx(30.68966, abs=1.0e-5)
    print("shmax_ll_95= ", shmax_ll_95)
    print("shmax_ll_20= ", shmax_ll_20)
    assert shmax_ul_95 == pytest.approx(91.68625, abs=1.0e-5)
    print("shmax_ul_95= ", shmax_ul_95)
    print("shmax_ul_20= ", shmax_ul_20)
    assert shmax_ll_20 < shmax_ul_20
    assert shmax_ll_20 > shmax_ll_95
