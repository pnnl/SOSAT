"""The State of Stress Analysis Tool (SOSAT) is a Python package that
helps analyze the state of stress in the subsurface using various types
of commonly available characterization data such as well logs, well
test data such as leakoff and minifrac tests, regional geologic
information, and constraints on the state of stress imposed by the
existence of faults and fractures with limited frictional shear
strength. It employs a Bayesian approach to integrate these data into
a probability density function for the principal stress components.
"""

__author__ = """Jeff Burghardt"""
__email__ = 'jeffrey.burghardt@pnnl.gov'
__version__ = '0.1.0'

from .stress_state import StressState
from .constraints import FaultConstraint
