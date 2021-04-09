import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pint
units = pint.UnitRegistry()
Q_ = units.Quantity
# Silence NEP 18 warning


"""Main module."""

gravity = 9.81 * units('m/s^2')


def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)
