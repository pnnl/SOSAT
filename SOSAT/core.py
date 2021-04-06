import numpy as np
import numpy.ma as ma
import pint
units = pint.UnitRegistry()
Q_ = units.Quantity
# Silence NEP 18 warning


"""Main module."""

gravity = 9.81 * units('m/s^2')
