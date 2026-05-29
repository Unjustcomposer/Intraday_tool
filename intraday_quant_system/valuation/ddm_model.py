"""
Dividend Discount Model (DDM)
==============================

Single-stage (Gordon Growth) and multi-stage DDM for Indian
equity valuation. Includes CAGR-based dividend growth estimation
and payout sustainability checks.

Non-dividend-paying stocks return ``None`` / ``NaN`` rather than
producing misleading valuations.

Usage::

    ddm = DDMModel()
    result = ddm.full_valuation(fundamentals_dict)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Indian-market defaults
# ---------------------------------------------------------------------------
_DEFAULT_RISK_FREE_RATE: float = 0.071
_DEFAULT_EQUITY_RISK_PREMIUM: float = 0.065
_DEFAULT_HIGH_GROWTH_YEARS: int = 5


class DDMModel:
    """Dividend Discount Model valuation engine.

    Supports:
        * Single-stage **Gordon Growth** model.
        * **Multi-stage** DDM with high-growth → stable transition.
        * **CAGR-based** dividend growth estimation from historical DPS.
        * **Payout sustainability** analysis (FCF coverage).
    """

    # --------------------------------------------------------------------- #
    # Gordon Growth (single stage)
    # --------------------------------------------------------------------- #

    @staticmethod
    def gordon_growth(
        dividend_per_share: float,
        cost_of_equity: float,
        growth_rate: float,
    ) -> Optional[float]:
        """Single-stage Gordon Growth intrinsic value.

        .. math::

            V_0 = \\frac{D_1}{k_e - g} = \\frac{D_0 (1+g)}{k_e - g}

        Args:
            dividend_per_share: Most recent annual DPS (₹).
            cost_of_equity: Required return on equity (decimal).
            growth_rate: Perpetual dividend growth rate (decimal).

        Returns:
            Fair value per share (₹), or ``None`` if:
                - DPS ≤ 0 (non-dividend payer)
                - cost_of_equity ≤ growth_rate (no finite value)
        """
        if dividend_per_share <= 0:
            logger.info("DPS ≤ 0; Gordon Growth not applicable.")
            return None

        if cost_of_equity <= growth_rate:
            logger.warning(
                "Cost of equity (%.4f) ≤ growth (%.4f); "
                "Gordon Growth diverges.",
                cost_of_equity, growth_rate,
            )
            return None

        d1 = dividend_per_share * (1.0 + growth_rate)
        value = d1 / (cost_of_equity - growth_rate)
        logger.debug("Gordon Growth value = ₹%.2f", value)
        return round(value, 2)

    # --------------------------------------------------------------------- #
    # Multi-stage DDM
    # --------------------------------------------------------------------- #

    @staticmethod
    def multi_stage_ddm(
        current_dps: float,
        high_growth_rate: float,
        stable_growth_rate: float,
        cost_of_equity: float,
        high_growth_years: int = _DEFAULT_HIGH_GROWTH_YEARS,
    ) -> Optional[float]:
        """Two-stage Dividend Discount Model.

        Phase 1: *high_growth_years* at *high_growth_rate*.
        Phase 2: perpetuity at *stable_growth_rate* (Gordon Growth).

        Args:
            current_dps: Current annual DPS (D₀).
            high_growth_rate: Growth rate during high-growth phase.
            stable_growth_rate: Long-run sustainable growth rate.
            cost_of_equity: Required rate of return.
            high_growth_years: Duration of the high-growth phase.

        Returns:
            Intrinsic value per share (₹), or ``None`` if DPS ≤ 0 or
            cost_of_equity ≤ stable_growth_rate.
        """
        if current_dps <= 0:
            logger.info("DPS ≤ 0; multi-stage DDM not applicable.")
            return None

        if cost_of_equity <= stable_growth_rate:
            logger.warning(
                "Cost of equity (%.4f) ≤ stable growth (%.4f); "
                "terminal value diverges.",
                cost_of_equity, stable_growth_rate,
            )
            return None

        # Phase 1: PV of high-growth dividends
        pv_high_growth = 0.0
        dps = current_dps
        for yr in range(1, high_growth_years + 1):
            dps *= (1.0 + high_growth_rate)
            pv_high_growth += dps / (1.0 + cost_of_equity) ** yr

        # Phase 2: terminal value at end of high-growth phase
        terminal_dps = dps * (1.0 + stable_growth_rate)
        terminal_value = terminal_dps / (cost_of_equity - stable_growth_rate)
        pv_terminal = terminal_value / (1.0 + cost_of_equity) ** high_growth_years

        intrinsic = pv_high_growth + pv_terminal
        logger.debug(
            "Multi-stage DDM: PV(high)=%.2f, PV(TV)=%.2f, total=%.2f",
            pv_high_growth, pv_terminal, intrinsic,
        )
        return round(intrinsic, 2)

    # --------------------------------------------------------------------- #
    # Growth estimation
    # --------------------------------------------------------------------- #

    @staticmethod
    def estimate_growth_rate(historical_dps_list: List[float]) -> Optional[float]:
        """Estimate dividend growth rate via CAGR.

        Uses the earliest and latest non-zero DPS values to compute
        the compound annual growth rate.

        Args:
            historical_dps_list: Chronological list of annual DPS values
                (earliest first).

        Returns:
            Annualised CAGR (decimal), or ``None`` if insufficient data.
        """
        if not historical_dps_list or len(historical_dps_list) < 2:
            logger.warning("Insufficient DPS history for growth estimation.")
            return None

        # Filter to positive values only
        positive = [(i, d) for i, d in enumerate(historical_dps_list) if d > 0]
        if len(positive) < 2:
            logger.warning("Fewer than 2 positive DPS observations.")
            return None

        first_idx, first_dps = positive[0]
        last_idx, last_dps = positive[-1]
        years = last_idx - first_idx
        if years <= 0:
            return None

        cagr = (last_dps / first_dps) ** (1.0 / years) - 1.0
        logger.debug(
            "DPS CAGR over %d years: %.4f (%.2f → %.2f)",
            years, cagr, first_dps, last_dps,
        )
        return round(cagr, 6)

    # --------------------------------------------------------------------- #
    # Payout sustainability
    # --------------------------------------------------------------------- #

    @staticmethod
    def payout_sustainability(
        fcf_per_share: float,
        dps: float,
    ) -> Dict[str, Any]:
        """Assess dividend payout sustainability.

        Args:
            fcf_per_share: Free cash flow per share (₹).
            dps: Dividend per share (₹).

        Returns:
            Dict with:
                ``coverage_ratio`` – FCF / DPS (>1 = sustainable).
                ``sustainable``    – bool flag.
                ``payout_ratio``   – DPS / FCF.
        """
        if dps <= 0:
            return {
                "coverage_ratio": None,
                "sustainable": None,
                "payout_ratio": None,
                "note": "No dividend paid.",
            }

        if fcf_per_share <= 0:
            return {
                "coverage_ratio": 0.0,
                "sustainable": False,
                "payout_ratio": None,
                "note": "Negative/zero FCF; dividend funded by debt or reserves.",
            }

        coverage = round(fcf_per_share / dps, 4)
        payout = round(dps / fcf_per_share, 4)
        sustainable = coverage >= 1.0

        logger.debug(
            "Payout sustainability: coverage=%.2f, payout=%.2f, sustainable=%s",
            coverage, payout, sustainable,
        )
        return {
            "coverage_ratio": coverage,
            "sustainable": sustainable,
            "payout_ratio": payout,
        }

    # --------------------------------------------------------------------- #
    # Full valuation
    # --------------------------------------------------------------------- #

    def full_valuation(
        self,
        fundamentals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """End-to-end DDM valuation.

        Expected keys in *fundamentals*:
            - ``dividend_per_share`` (float): Most recent annual DPS.
            - ``historical_dps`` (list[float], optional): Chronological DPS.
            - ``beta`` (float, optional): Equity beta, default 1.0.
            - ``risk_free_rate`` (float, optional): Override G-Sec yield.
            - ``equity_risk_premium`` (float, optional): Override ERP.
            - ``high_growth_rate`` (float, optional): High-growth phase rate.
            - ``stable_growth_rate`` (float, optional): Terminal growth.
            - ``fcf_per_share`` (float, optional): For sustainability check.

        Returns:
            Dict with ``fair_value``, ``gordon_value``, ``multi_stage_value``,
            ``growth_estimate``, ``sustainability``, ``assumptions``.
            Values are ``None``/``NaN`` for non-dividend-paying stocks.
        """
        symbol = fundamentals.get("symbol", "UNKNOWN")
        logger.info("Full DDM valuation for %s", symbol)

        dps = fundamentals.get("dividend_per_share", 0.0)
        historical_dps = fundamentals.get("historical_dps", [])
        beta = fundamentals.get("beta", 1.0)
        rfr = fundamentals.get("risk_free_rate", _DEFAULT_RISK_FREE_RATE)
        erp = fundamentals.get("equity_risk_premium", _DEFAULT_EQUITY_RISK_PREMIUM)
        fcf_ps = fundamentals.get("fcf_per_share", 0.0)

        # Cost of equity via CAPM
        cost_of_equity = rfr + max(beta, 0.0) * erp

        # Estimate growth from history, with fallback
        growth_estimate = self.estimate_growth_rate(historical_dps)
        stable_g = fundamentals.get(
            "stable_growth_rate",
            min(growth_estimate, 0.06) if growth_estimate is not None else 0.05,
        )
        high_g = fundamentals.get(
            "high_growth_rate",
            min(growth_estimate * 1.5, 0.20) if growth_estimate is not None else 0.12,
        )

        # Guard: cap growth rates to sensible bounds
        stable_g = min(stable_g, cost_of_equity - 0.005)
        high_g = max(high_g, stable_g)

        # ---- Non-dividend payer fast path ---------------------------------
        if dps <= 0:
            logger.info(
                "%s does not pay dividends; DDM returns None.", symbol,
            )
            return {
                "fair_value": None,
                "gordon_value": None,
                "multi_stage_value": None,
                "growth_estimate": growth_estimate,
                "sustainability": {
                    "coverage_ratio": None,
                    "sustainable": None,
                    "payout_ratio": None,
                    "note": "Non-dividend-paying stock.",
                },
                "assumptions": {
                    "dps": dps,
                    "cost_of_equity": round(cost_of_equity, 6),
                    "stable_growth_rate": stable_g,
                    "high_growth_rate": high_g,
                },
            }

        # ---- Valuations ---------------------------------------------------
        gordon_val = self.gordon_growth(dps, cost_of_equity, stable_g)
        multi_val = self.multi_stage_ddm(
            dps, high_g, stable_g, cost_of_equity,
            high_growth_years=fundamentals.get(
                "high_growth_years", _DEFAULT_HIGH_GROWTH_YEARS,
            ),
        )

        # Use multi-stage as primary; Gordon as fallback
        fair_value = multi_val if multi_val is not None else gordon_val

        sustainability = self.payout_sustainability(fcf_ps, dps)

        assumptions = {
            "dps": dps,
            "cost_of_equity": round(cost_of_equity, 6),
            "stable_growth_rate": stable_g,
            "high_growth_rate": high_g,
            "beta": beta,
            "risk_free_rate": rfr,
            "equity_risk_premium": erp,
        }

        logger.info("DDM fair value for %s = %s", symbol, fair_value)
        return {
            "fair_value": fair_value,
            "gordon_value": gordon_val,
            "multi_stage_value": multi_val,
            "growth_estimate": growth_estimate,
            "sustainability": sustainability,
            "assumptions": assumptions,
        }
