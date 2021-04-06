# import pytest
import pytest
from SOSAT import StressState


def test_constructor():
    ss = StressState(1.0,
                     2.5,
                     0.3,
                     depth_unit='km',
                     density_unit='g/cm^3',
                     pressure_unit='MPa')

    sigv = ss.vertical_stress
    sigv == pytest.approx(24.525)
