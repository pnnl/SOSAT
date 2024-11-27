import pytest
import numpy as np
import matplotlib.pyplot as plt

from SOSAT import StressState
from SOSAT.constraints import FaultConstraint


def test_PDF_plotting():

    # depth in meters
    depth = 1228.3
    # density in kg/m^3
    avg_overburden_density = 2580.0
    # pore pressure gradient in MPa/km
    pore_pressure_grad = 9.955

    pore_pressure = pore_pressure_grad * (1.0 / 1000) * depth

    # Create two StessState objects that differ only for nbins
    ss_50 = StressState(depth=depth,
                        avg_overburden_density=avg_overburden_density,
                        pore_pressure=pore_pressure, nbins=50)

    ss_5000 = StressState(depth=depth,
                          avg_overburden_density=avg_overburden_density,
                          pore_pressure=pore_pressure, nbins=5000)

    fc = FaultConstraint()

    ss_50.add_constraint(fc)
    ss_5000.add_constraint(fc)

    ss_50.evaluate_posterior()
    ss_5000.evaluate_posterior()

    # These lines are repeated in StressState.plot_posterior()
    # Convert data format to bin-centered eval of PDF for plotting
    dsig_50 = ss_50.shmax_grid[0][1] - ss_50.shmax_grid[0][0]
    posterior_PDF_50 = ss_50.posterior / dsig_50**2.0

    dsig_5000 = ss_5000.shmax_grid[0][1] - ss_5000.shmax_grid[0][0]
    posterior_PDF_5000 = ss_5000.posterior / dsig_5000**2.0

    # Test if maximum value of posterior is approximately equal
    # Small err is expected due to different discretizations of sigma-space
    tol = 5.0e-2
    rel_err = (np.nanmax(posterior_PDF_50) - np.nanmax(posterior_PDF_5000)) \
        / np.nanmax(posterior_PDF_50)
    assert rel_err <= tol

    fig_50 = ss_50.plot_posterior()
    fig_5000 = ss_5000.plot_posterior()

    fig_50.savefig("50_bin_posterior_PDF.png")
    fig_5000.savefig("5000_bin_posterior_PDF.png")
