"""
DCF (Discounted Cash Flow) Valuation Engine
============================================

Two-stage DCF model designed for NSE-listed Indian equities.

Defaults:
    - Risk-free rate  : India 10-Year G-Sec yield (~7.1 %)
    - Equity risk premium : 6.5 % (Damodaran estimate for India)
    - High-growth phase : 5 years
    - Fade / transition  : 5 years

Usage::

    dcf = DCFModel()
    result = dcf.full_valuation(fundamentals_dict)
    print(result['fair_value'])
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Indian-market defaults
# ---------------------------------------------------------------------------
_DEFAULT_RISK_FREE_RATE: float = 0.071        # India 10Y G-Sec yield
_DEFAULT_EQUITY_RISK_PREMIUM: float = 0.065   # Damodaran India ERP
_DEFAULT_HIGH_GROWTH_YEARS: int = 5
_DEFAULT_FADE_YEARS: int = 5
_DEFAULT_TERMINAL_GROWTH: float = 0.05        # Long-run nominal GDP growth


class DCFModel:
    """Discounted Cash Flow valuation engine for Indian equities.

    The model performs a two-stage (high-growth → fade → terminal) DCF,
    computes per-share intrinsic value, and provides a sensitivity matrix
    over growth-rate / WACC ranges.
    """

    # --------------------------------------------------------------------- #
    # Core calculations
    # --------------------------------------------------------------------- #

    @staticmethod
    def calculate_wacc(
        beta: float,
        risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
        equity_risk_premium: float = _DEFAULT_EQUITY_RISK_PREMIUM,
        debt_cost: float = 0.09,
        tax_rate: float = 0.2508,
        debt_ratio: float = 0.0,
    ) -> float:
        """Weighted Average Cost of Capital.

        Args:
            beta: Equity beta relative to Nifty 50.
            risk_free_rate: Yield on India 10Y G-Sec.
            equity_risk_premium: Expected return above risk-free.
            debt_cost: Pre-tax cost of debt.
            tax_rate: Effective corporate tax rate (new regime ~25.17 %).
            debt_ratio: Debt / (Debt + Equity) ratio.

        Returns:
            WACC as a decimal (e.g. 0.12 for 12 %).
        """
        if beta < 0:
            logger.warning("Negative beta (%.2f) received; clamping to 0.", beta)
            beta = 0.0

        cost_of_equity = risk_free_rate + beta * equity_risk_premium
        equity_weight = 1.0 - debt_ratio
        debt_weight = debt_ratio

        wacc = (
            equity_weight * cost_of_equity
            + debt_weight * debt_cost * (1.0 - tax_rate)
        )
        logger.debug(
            "WACC=%.4f  (Ke=%.4f, beta=%.2f, Kd=%.4f, D/V=%.2f)",
            wacc, cost_of_equity, beta, debt_cost, debt_ratio,
        )
        return wacc

    @staticmethod
    def project_fcf(
        base_fcf: float,
        high_growth_rate: float,
        stable_growth_rate: float,
        high_growth_years: int = _DEFAULT_HIGH_GROWTH_YEARS,
        fade_years: int = _DEFAULT_FADE_YEARS,
    ) -> List[float]:
        """Project Free Cash Flows over a two-stage horizon.

        Phase 1 – *high_growth_years* at *high_growth_rate*.
        Phase 2 – *fade_years* linearly fading from high to stable growth.

        Args:
            base_fcf: Most recent trailing-twelve-month FCF (₹ Cr).
            high_growth_rate: Expected growth in high-growth phase.
            stable_growth_rate: Long-run sustainable growth rate.
            high_growth_years: Duration of high-growth phase.
            fade_years: Duration of transition phase.

        Returns:
            List of projected FCFs for each year.
        """
        if base_fcf <= 0:
            logger.warning(
                "Base FCF is non-positive (%.2f); projections may be meaningless.",
                base_fcf,
            )

        projected: List[float] = []
        current_fcf = base_fcf

        # Phase 1: high growth
        for yr in range(1, high_growth_years + 1):
            current_fcf *= (1.0 + high_growth_rate)
            projected.append(round(current_fcf, 4))

        # Phase 2: linear fade
        if fade_years > 0:
            step = (high_growth_rate - stable_growth_rate) / fade_years
            fade_rate = high_growth_rate
            for yr in range(1, fade_years + 1):
                fade_rate -= step
                current_fcf *= (1.0 + fade_rate)
                projected.append(round(current_fcf, 4))

        logger.debug(
            "Projected %d years of FCF (high=%.2f%%, stable=%.2f%%)",
            len(projected),
            high_growth_rate * 100,
            stable_growth_rate * 100,
        )
        return projected

    @staticmethod
    def terminal_value(
        last_fcf: float,
        terminal_growth_rate: float,
        wacc: float,
    ) -> float:
        """Gordon Growth terminal value.

        Args:
            last_fcf: FCF in the final explicit-forecast year.
            terminal_growth_rate: Perpetual growth rate (≤ nominal GDP).
            wacc: Weighted average cost of capital.

        Returns:
            Terminal value (₹ Cr).

        Raises:
            ValueError: If WACC ≤ terminal growth (no finite TV).
        """
        if wacc <= terminal_growth_rate:
            raise ValueError(
                f"WACC ({wacc:.4f}) must exceed terminal growth "
                f"({terminal_growth_rate:.4f}) for a finite terminal value."
            )
        tv = last_fcf * (1.0 + terminal_growth_rate) / (wacc - terminal_growth_rate)
        logger.debug("Terminal value = %.2f", tv)
        return tv

    @staticmethod
    def intrinsic_value(
        projected_fcfs: List[float],
        terminal_value: float,
        wacc: float,
        shares_outstanding: float,
    ) -> float:
        """Per-share intrinsic value from DCF.

        Discounts each projected FCF and the terminal value back to present,
        then divides by shares outstanding.

        Args:
            projected_fcfs: Annual projected FCFs.
            terminal_value: Terminal value at end of projection horizon.
            wacc: Discount rate.
            shares_outstanding: Diluted shares outstanding (Cr or absolute).

        Returns:
            Intrinsic value per share (₹).

        Raises:
            ValueError: If shares_outstanding ≤ 0.
        """
        if shares_outstanding <= 0:
            raise ValueError("shares_outstanding must be positive.")

        pv_fcfs = sum(
            fcf / (1.0 + wacc) ** yr
            for yr, fcf in enumerate(projected_fcfs, start=1)
        )
        pv_tv = terminal_value / (1.0 + wacc) ** len(projected_fcfs)
        enterprise_value = pv_fcfs + pv_tv
        per_share = enterprise_value / shares_outstanding

        logger.debug(
            "PV(FCFs)=%.2f  PV(TV)=%.2f  EV=%.2f  per_share=%.2f",
            pv_fcfs, pv_tv, enterprise_value, per_share,
        )
        return round(per_share, 2)

    # --------------------------------------------------------------------- #
    # Sensitivity analysis
    # --------------------------------------------------------------------- #

    def sensitivity_analysis(
        self,
        base_fcf: float,
        shares_outstanding: float,
        growth_range: List[float],
        wacc_range: List[float],
        terminal_growth_rate: float = _DEFAULT_TERMINAL_GROWTH,
        high_growth_years: int = _DEFAULT_HIGH_GROWTH_YEARS,
        fade_years: int = _DEFAULT_FADE_YEARS,
    ) -> pd.DataFrame:
        """Two-dimensional sensitivity matrix: growth × WACC.

        Args:
            base_fcf: Base free cash flow.
            shares_outstanding: Diluted share count.
            growth_range: List of high-growth-rate assumptions.
            wacc_range: List of WACC assumptions.
            terminal_growth_rate: Long-run growth for TV.
            high_growth_years: Years in high-growth phase.
            fade_years: Years in fade phase.

        Returns:
            DataFrame with growth rates as index, WACCs as columns,
            and intrinsic-value per share as cell values.
        """
        logger.info(
            "Running sensitivity: %d growth × %d WACC scenarios",
            len(growth_range), len(wacc_range),
        )
        matrix: Dict[float, Dict[float, Optional[float]]] = {}

        for g in growth_range:
            row: Dict[float, Optional[float]] = {}
            for w in wacc_range:
                try:
                    fcfs = self.project_fcf(
                        base_fcf, g, terminal_growth_rate,
                        high_growth_years, fade_years,
                    )
                    tv = self.terminal_value(fcfs[-1], terminal_growth_rate, w)
                    iv = self.intrinsic_value(fcfs, tv, w, shares_outstanding)
                    row[w] = iv
                except (ValueError, ZeroDivisionError) as exc:
                    logger.warning("Sensitivity (g=%.3f, w=%.3f): %s", g, w, exc)
                    row[w] = None
            matrix[g] = row

        df = pd.DataFrame(matrix).T
        df.index.name = "Growth Rate"
        df.columns.name = "WACC"
        return df

    # --------------------------------------------------------------------- #
    # Full valuation convenience wrapper
    # --------------------------------------------------------------------- #

    def full_valuation(
        self,
        fundamentals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """End-to-end DCF valuation from a fundamentals dictionary.

        Expected keys in *fundamentals*:
            - ``free_cash_flow`` (float): Trailing 12-month FCF (₹ Cr).
            - ``shares_outstanding`` (float): Diluted share count.
            - ``beta`` (float): Equity beta.
            - ``debt_ratio`` (float, optional): D/(D+E), default 0.
            - ``debt_cost`` (float, optional): Pre-tax cost of debt.
            - ``tax_rate`` (float, optional): Effective tax rate.
            - ``high_growth_rate`` (float, optional): Expected high-growth rate.
            - ``stable_growth_rate`` (float, optional): Terminal growth rate.
            - ``risk_free_rate`` (float, optional): Override for G-Sec yield.
            - ``equity_risk_premium`` (float, optional): Override for ERP.

        Returns:
            Dict with keys:
                ``fair_value``          – per-share intrinsic value (₹).
                ``sensitivity_matrix``  – DataFrame of growth × WACC.
                ``assumptions``         – dict of all inputs used.
                ``enterprise_value``    – total enterprise value.
                ``wacc``                – computed WACC.
        """
        logger.info("Running full DCF valuation")

        # ---- Extract / default parameters --------------------------------
        base_fcf: float = fundamentals.get("free_cash_flow", 0.0)
        shares: float = fundamentals.get("shares_outstanding", 1.0)
        beta: float = fundamentals.get("beta", 1.0)
        debt_ratio: float = fundamentals.get("debt_ratio", 0.0)
        debt_cost: float = fundamentals.get("debt_cost", 0.09)
        tax_rate: float = fundamentals.get("tax_rate", 0.2508)
        rfr: float = fundamentals.get("risk_free_rate", _DEFAULT_RISK_FREE_RATE)
        erp: float = fundamentals.get("equity_risk_premium", _DEFAULT_EQUITY_RISK_PREMIUM)
        high_g: float = fundamentals.get("high_growth_rate", 0.15)
        stable_g: float = fundamentals.get("stable_growth_rate", _DEFAULT_TERMINAL_GROWTH)
        high_years: int = fundamentals.get("high_growth_years", _DEFAULT_HIGH_GROWTH_YEARS)
        fade_years: int = fundamentals.get("fade_years", _DEFAULT_FADE_YEARS)

        # ---- Guard against obviously bad inputs --------------------------
        if base_fcf <= 0:
            logger.warning(
                "Non-positive FCF (%.2f); DCF value will be unreliable.", base_fcf,
            )
        if shares <= 0:
            logger.error("shares_outstanding must be > 0; returning NaN.")
            return {
                "fair_value": float("nan"),
                "sensitivity_matrix": pd.DataFrame(),
                "assumptions": {},
                "enterprise_value": float("nan"),
                "wacc": float("nan"),
            }

        # ---- Compute WACC ------------------------------------------------
        wacc = self.calculate_wacc(
            beta=beta,
            risk_free_rate=rfr,
            equity_risk_premium=erp,
            debt_cost=debt_cost,
            tax_rate=tax_rate,
            debt_ratio=debt_ratio,
        )

        # ---- Project FCFs & terminal value -------------------------------
        projected = self.project_fcf(
            base_fcf, high_g, stable_g, high_years, fade_years,
        )
        try:
            tv = self.terminal_value(projected[-1], stable_g, wacc)
        except ValueError as exc:
            logger.error("Terminal value calculation failed: %s", exc)
            return {
                "fair_value": float("nan"),
                "sensitivity_matrix": pd.DataFrame(),
                "assumptions": {"error": str(exc)},
                "enterprise_value": float("nan"),
                "wacc": wacc,
            }

        fair_value = self.intrinsic_value(projected, tv, wacc, shares)

        # ---- Enterprise value (un-divided) --------------------------------
        pv_fcfs = sum(
            fcf / (1.0 + wacc) ** yr
            for yr, fcf in enumerate(projected, start=1)
        )
        pv_tv = tv / (1.0 + wacc) ** len(projected)
        enterprise_value = round(pv_fcfs + pv_tv, 2)

        # ---- Sensitivity matrix ------------------------------------------
        growth_range = [
            round(high_g + delta, 4)
            for delta in np.arange(-0.05, 0.06, 0.025)
        ]
        wacc_range = [
            round(wacc + delta, 4)
            for delta in np.arange(-0.02, 0.025, 0.01)
        ]
        sensitivity = self.sensitivity_analysis(
            base_fcf, shares, growth_range, wacc_range,
            terminal_growth_rate=stable_g,
            high_growth_years=high_years,
            fade_years=fade_years,
        )

        assumptions = {
            "base_fcf": base_fcf,
            "shares_outstanding": shares,
            "beta": beta,
            "risk_free_rate": rfr,
            "equity_risk_premium": erp,
            "debt_cost": debt_cost,
            "tax_rate": tax_rate,
            "debt_ratio": debt_ratio,
            "high_growth_rate": high_g,
            "stable_growth_rate": stable_g,
            "high_growth_years": high_years,
            "fade_years": fade_years,
            "wacc": wacc,
        }

        logger.info("DCF fair value = ₹%.2f", fair_value)
        return {
            "fair_value": fair_value,
            "sensitivity_matrix": sensitivity,
            "assumptions": assumptions,
            "enterprise_value": enterprise_value,
            "wacc": wacc,
        }
