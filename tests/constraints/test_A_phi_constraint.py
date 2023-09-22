import pytest
from scipy.stats import truncnorm
import numpy as np
import matplotlib.pyplot as plt
from numpy import unravel_index
from SOSAT import StressState
from SOSAT.constraints import FaultingRegimeAphiConstraint


def test_A_phi():
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

    # add faulting regime Aphi constraint
    lower_bound = 0
    upper_bound = 3
    mean = 1.5  # (changeable, mean of A_phi value at this region)
    variance = 0.08  # (changable, variance of A_phi value at this region)
    std = np.sqrt(variance)
    rv = truncnorm((lower_bound - mean) / std, (upper_bound - mean) / std,
                   mean, std)
    # check random variable: plot pdf and cdf of the defined random variable
    x = np.linspace(0, 3, 50)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Mean is {mean:.2f}; Std is {std:.2f}')
    ax[0].plot(x, rv.pdf(x), 'k-', lw=2)
    ax[0].set_xlabel(r'$A_{\phi}$')
    ax[0].set_ylabel('pdf')
    ax[1].plot(x, rv.cdf(x), 'k-', lw=2)
    ax[1].set_xlabel(r'$A_{\phi}$')
    ax[1].set_ylabel('cdf')

    frAc = FaultingRegimeAphiConstraint(rv)

    ss.add_constraint(frAc)

    fig2 = ss.plot_posterior()
    fig2.savefig("fault_regime_constraint_posterior.png")

    # perform test
    max_post_index = unravel_index(ss.posterior.argmax(), ss.posterior.shape)
    max_Aphi = ss.A_phi_calculate()[max_post_index[0], max_post_index[1]]
    max_Aphi == pytest.approx(mean)


if __name__ == 'main':
    test_A_phi()
