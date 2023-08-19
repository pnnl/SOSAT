import pytest
import numpy as np
from scipy.stats import uniform, weibull_min
import matplotlib.pyplot as plt

from SOSAT import StressState
from SOSAT.constraints import DITFConstraint
from SOSAT.constraints import FaultConstraint
from SOSAT.constraints import StressMeasurement
from os import path


def test_DITF():
    # depth in ft
    depth = 8480.0
    # density in g/cm^3
    avg_overburden_density = 2.58
    # pore pressure gradient in psi/ft
    pore_pressure_grad = 0.43

    pore_pressure = pore_pressure_grad * depth

    ss = StressState(depth=depth,
                     avg_overburden_density=avg_overburden_density,
                     pore_pressure=pore_pressure,
                     density_unit='g/cm^3',
                     stress_unit='psi',
                     pressure_unit='psi',
                     depth_unit='ft',
                     nbins=500,
                     min_stress_ratio=0.4,
                     max_stress_ratio=2.5)

    '''
    probability density for the minimum tensile strength over
    the interval
    '''
    tensile_strength_dist = uniform(loc=999.0, scale=1.0)

    ts = np.linspace(0.0, 5000.0, 10000)
    pts = tensile_strength_dist.pdf(ts)
    ts_fig = plt.figure()
    ax_ts = ts_fig.add_subplot(111)
    ax_ts.plot(ts, pts, "k")
    ax_ts.set_xlabel("MinimumTensile Strength (psi)")
    ax_ts.set_ylabel("Probability Density")
    ts_fig.savefig("Tensile_Strength_PDF.png")

    mud_pressure_upper = 8.3 * 0.052 * depth + 101.0
    mud_pressure_lower = 8.3 * 0.052 * depth + 100.0
    print("mud pressure upper = ", mud_pressure_upper)
    print(mud_pressure_upper / depth, " psi/ft")

    print("mud pressure lower = ", mud_pressure_lower)
    print(mud_pressure_lower / depth, " psi/ft")
    mud_pressure_dist = uniform(loc=mud_pressure_lower,
                                scale=(mud_pressure_upper
                                       - mud_pressure_lower))

    mud_temperature_upper = 105.0
    mud_temperature_lower = 104.0
    mud_temperature_dist = uniform(loc=mud_temperature_lower,
                                   scale=(mud_temperature_upper
                                          - mud_temperature_lower))

    mud_temperature_array = np.linspace(100, 130, 500)
    mud_temperature_fig = plt.figure()
    mud_temperature_ax = mud_temperature_fig.add_subplot(111)
    mud_temperature_ax.plot(mud_temperature_array,
                            mud_temperature_dist.pdf(mud_temperature_array),
                            'k')
    mud_temperature_ax.set_xlabel("Minimum Drilling Mud temperature (F)")
    mud_temperature_ax.set_ylabel("PDF")
    mud_temperature_fig.savefig("Mud_temperature_PDF.png")

    formation_temperature = 392.0

    fault_friction_dist = uniform(0.59, scale=0.1)
    shmin_lower = 6020.0
    shmin_upper = 6699.0
    shmin_dist = uniform(loc=shmin_lower,
                         scale=(shmin_upper - shmin_lower))
    smc = StressMeasurement(shmin_dist=shmin_dist)

    # YM in psi
    YM = 6.0e6
    PR = 0.25
    CTE = 2.75e-6
    DITFc = DITFConstraint(DITF_exists=True,
                           mud_pressure_dist=mud_pressure_dist,
                           mud_temperature_dist=mud_temperature_dist,
                           tensile_strength_dist=tensile_strength_dist,
                           formation_temperature=formation_temperature,
                           YM=YM,
                           PR=PR,
                           CTE=CTE,
                           pressure_unit='psi')
    ss.add_constraint(FaultConstraint(fault_friction_dist))
    ss.add_constraint(DITFc)
    ss.add_constraint(smc)
    fig = ss.plot_posterior()
    fig.savefig("DITF_constraint_posterior.png")

    sig_theta_nominal = 3.0 * shmin_lower - 2.0 * pore_pressure

    deltaP = mud_pressure_upper - pore_pressure
    deltaT = formation_temperature - mud_temperature_lower
    sig_thermal = CTE * YM * deltaT / (1.0 - PR)
    shmax_ll_expect = sig_theta_nominal - deltaP - sig_thermal + 1000.0
    print("expected shmax ll= ", shmax_ll_expect)
    shmax_ll, shmax_ul = ss.get_shmax_confidence_intervals(0.95)
    print("shmax_ll= ", shmax_ll)
    shmax_ll == pytest.approx(6710.1)
    print("shmax_ul= ", shmax_ul)
    shmax_ul == pytest.approx(12699.6)


if __name__ == '__main__':
    test_DITF()
