"""
Reconciliation_HTS - External library for hierarchical time series reconciliation.
Source: [https://github.com/Adrien-df/reconciliation_hts.git]
Integrated into demandprediction for internal use.
"""
from ._version import __version__
from .reconciliation import *

__all__ = ["reconciliation", "__version__"]