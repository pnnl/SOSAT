import pytest
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

from SOSAT import StressState
from SOSAT.constraints import FaultConstraint


def test_marginals():
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

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    shmin, shmin_cdf = ss.get_shmin_marginal_cdf()
    ax3.plot(shmin, shmin_cdf, "k")
    ax3.set_xlabel("Minimum Horizontal Stress")
    ax3.set_ylabel("Cumulative Probability Density")
    fig3.savefig("fault_constraint_shmin_cdf.png")

    shmin_ll, shmin_ul = ss.get_shmin_confidence_intervals(0.95)
    print("shmin_ll= ", shmin_ll)
    shmin_ll == pytest.approx(21.785)
    print("shmin_ul= ", shmin_ul)
    shmin_ul == pytest.approx(73.877)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    shmin, shmin_cdf = ss.get_shmax_marginal_cdf()
    ax4.plot(shmin, shmin_cdf, "k")
    ax4.set_xlabel("Maximum Horizontal Stress")
    ax4.set_ylabel("Cumulative Probability Density")
    fig4.savefig("fault_constraint_shmax_cdf.png")

    shmax_ll, shmax_ul = ss.get_shmax_confidence_intervals(0.95)
    shmax_ll == pytest.approx(30.689)
    print("shmax_ll= ", shmax_ll)
    shmax_ul == pytest.approx(91.686)
    print("shmax_ul= ", shmax_ul)
