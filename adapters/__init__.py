"""
adapters/ — Cross-version adapter layer.

Integrates artifacts and code from earlier model versions (2.0, 2.1)
and external source repos (deep_sgdf_delta) into the 3.0 pipeline.
"""

from adapters.residual_p5m_adapter import (
    ResidualP5MAdapter,
    RESIDUAL_P5M_REAL_APPLIED,
    RESIDUAL_P5M_CATBOOST_APPLIED,
    RESIDUAL_P5M_CODE_ONLY,
    RESIDUAL_P5M_NO_OP,
)
from adapters.negative_classifier_adapter import (
    NegativeClassifierAdapter,
)

__all__ = [
    "ResidualP5MAdapter",
    "RESIDUAL_P5M_REAL_APPLIED",
    "RESIDUAL_P5M_CATBOOST_APPLIED",
    "RESIDUAL_P5M_CODE_ONLY",
    "RESIDUAL_P5M_NO_OP",
    "NegativeClassifierAdapter",
]
