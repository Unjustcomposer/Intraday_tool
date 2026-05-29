"""
Valuation Triangulation System
==============================

Multi-model valuation framework for Indian equities.
Combines DCF, Relative/Multiples, and Dividend Discount models
into a single triangulated fair-value estimate with confidence scoring.

Modules:
    dcf_model           – Discounted Cash Flow valuation engine
    relative_valuation  – Peer-comparable multiples-based valuation
    ddm_model           – Dividend Discount Model (Gordon Growth + multi-stage)
    triangulation_engine – Master engine that blends all three models
"""

from valuation.dcf_model import DCFModel
from valuation.relative_valuation import RelativeValuation
from valuation.ddm_model import DDMModel
from valuation.triangulation_engine import ValuationTriangulator

__all__ = [
    "DCFModel",
    "RelativeValuation",
    "DDMModel",
    "ValuationTriangulator",
]
