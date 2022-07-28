import pytest
import numpy as np
from scipy.stats import uniform, weibull_min
import matplotlib.pyplot as plt

from SOSAT import StressState
from SOSAT.constraints import DITFConstraint


def test_DITF():
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

    # probability density for the minimum tensile strength over
    # the interval
    tensile_strength_dist = weibull_min(c=1.1)

    ts = np.linspace(0.0, 10.0, 500)
    pts = tensile_strength_dist.pdf(ts)
    ts_fig = plt.figure()
    ax_ts = ts_fig.add_subplot(111)
    ax_ts.plot(ts, pts, "k")
    ax_ts.set_xlabel("MinimumTensile Strength (MPa)")
    ax_ts.set_ylabel("Probability Density")
    ts_fig.savefig("Tensile_Strength_PDF.png")

    mud_pressure_dist = uniform(loc=17.32, scale=(19.32 - 17.32))
    mud_temperature_dist = uniform(loc=35.0, scale=(50.0 - 35.0))
    formation_temperature = 30.7
    # YM in MPa, so 19,000 MPa = 19 GPa
    YM = 19.0e3
    PR = 0.25
    CTE = 2.4e-6
    DITFc = DITFConstraint(DITF_exists=False,
                           mud_pressure_dist=mud_pressure_dist,
                           mud_temperature_dist=mud_temperature_dist,
                           tensile_strength_dist=tensile_strength_dist,
                           formation_temperature=formation_temperature,
                           YM=YM,
                           PR=PR,
                           CTE=CTE,
                           pressure_unit='MPa')
    ss.add_constraint(DITFc)
    fig = ss.plot_posterior()
    fig.savefig("DITF_constraint_posterior.png")

    shmax_ll, shmax_ul = ss.get_shmax_confidence_intervals(0.95)
    print("shmax_ll= ", shmax_ll)
    shmax_ll == pytest.approx(34.2515)
    print("shmax_ul= ", shmax_ul)
    shmax_ul == pytest.approx(99.255)
