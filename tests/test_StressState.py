# import pytest
from SOSAT import StressState


def test_constructor():
    ss = StressState(1.0, 2.5, 'km', 'g/cm^3')

    sigv = ss.vertical_stress.to('MPa')
    assert sigv.magnitude == 24.525
