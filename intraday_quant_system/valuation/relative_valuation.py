"""
Relative / Multiples-Based Valuation Engine
=============================================

Computes valuation multiples (PE, PB, EV/EBITDA, EV/Sales, PEG),
compares them to sector peers via percentile ranks and z-scores,
and derives a fair-value range from peer-median multiples.

Sector taxonomy follows NSE / BSE classifications commonly used
for Indian equities.

Usage::

    rv = RelativeValuation()
    result = rv.full_valuation(fundamentals_dict, sector_data)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NSE Sector Classification
# ---------------------------------------------------------------------------
SECTOR_MAP: Dict[str, str] = {
    # Broad groups used for weight mapping in triangulation
    "IT": "IT",
    "TECHNOLOGY": "IT",
    "SOFTWARE": "IT",
    "BANKING": "BANKING",
    "FINANCE": "BANKING",
    "NBFC": "BANKING",
    "INSURANCE": "BANKING",
    "PHARMA": "PHARMA",
    "HEALTHCARE": "PHARMA",
    "AUTO": "AUTO",
    "AUTOMOBILE": "AUTO",
    "FMCG": "FMCG",
    "CONSUMER": "FMCG",
    "CONSUMER STAPLES": "FMCG",
    "CONSUMER DISCRETIONARY": "FMCG",
    "ENERGY": "ENERGY",
    "OIL & GAS": "ENERGY",
    "METALS": "METALS",
    "MINING": "METALS",
    "CEMENT": "CEMENT",
    "CONSTRUCTION": "CEMENT",
    "INFRASTRUCTURE": "CEMENT",
    "REALTY": "REALTY",
    "REAL ESTATE": "REALTY",
    "TELECOM": "TELECOM",
    "MEDIA": "TELECOM",
    "UTILITIES": "UTILITIES",
    "POWER": "UTILITIES",
    "CHEMICALS": "CHEMICALS",
    "TEXTILES": "TEXTILES",
}


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    """Return *numerator / denominator* or ``None`` when division is invalid."""
    if denominator is None or denominator == 0 or math.isnan(denominator):
        return None
    result = numerator / denominator
    if math.isinf(result) or math.isnan(result):
        return None
    return round(result, 4)


def _normalise_sector(raw: str) -> str:
    """Map a free-text sector name to a canonical group."""
    key = raw.strip().upper()
    return SECTOR_MAP.get(key, "OTHER")


class RelativeValuation:
    """Peer-comparable multiples-based valuation engine.

    Designed for NSE-listed equities with sector-aware comparisons.
    """

    # --------------------------------------------------------------------- #
    # Multiples computation
    # --------------------------------------------------------------------- #

    @staticmethod
    def compute_multiples(fundamentals: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Derive standard valuation multiples from a fundamentals dict.

        Expected keys:
            ``price``, ``eps``, ``book_value_per_share``,
            ``enterprise_value``, ``ebitda``, ``revenue``,
            ``earnings_growth`` (forward, decimal), ``dividend_per_share``.

        Returns:
            Dict with ``PE``, ``PB``, ``EV_EBITDA``, ``EV_Sales``, ``PEG``,
            ``dividend_yield``.  Values may be ``None`` when the metric
            cannot be computed (e.g. negative earnings).
        """
        price = fundamentals.get("price", 0.0)
        eps = fundamentals.get("eps", 0.0)
        bvps = fundamentals.get("book_value_per_share", 0.0)
        ev = fundamentals.get("enterprise_value", 0.0)
        ebitda = fundamentals.get("ebitda", 0.0)
        revenue = fundamentals.get("revenue", 0.0)
        eg = fundamentals.get("earnings_growth", 0.0)
        dps = fundamentals.get("dividend_per_share", 0.0)

        # P/E  — only meaningful when eps > 0
        pe = _safe_divide(price, eps) if eps > 0 else None

        # P/B  — only meaningful when bvps > 0
        pb = _safe_divide(price, bvps) if bvps > 0 else None

        # EV/EBITDA — only meaningful when EBITDA > 0
        ev_ebitda = _safe_divide(ev, ebitda) if ebitda > 0 else None

        # EV/Sales
        ev_sales = _safe_divide(ev, revenue) if revenue > 0 else None

        # PEG  — needs both positive PE and positive growth
        peg: Optional[float] = None
        if pe is not None and eg and eg > 0:
            peg = _safe_divide(pe, eg * 100)  # growth in %

        # Dividend Yield
        div_yield = _safe_divide(dps, price) if price > 0 and dps > 0 else None

        multiples = {
            "PE": pe,
            "PB": pb,
            "EV_EBITDA": ev_ebitda,
            "EV_Sales": ev_sales,
            "PEG": peg,
            "dividend_yield": div_yield,
        }
        logger.debug("Computed multiples: %s", multiples)
        return multiples

    # --------------------------------------------------------------------- #
    # Sector comparison
    # --------------------------------------------------------------------- #

    @staticmethod
    def sector_comparison(
        symbol: str,
        multiples: Dict[str, Optional[float]],
        sector_data: Dict[str, Dict[str, Optional[float]]],
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Compare a stock's multiples against sector peers.

        Args:
            symbol: Ticker being valued.
            multiples: Output of :meth:`compute_multiples` for *symbol*.
            sector_data: ``{peer_symbol: {PE, PB, …}}`` for all sector
                peers (including *symbol* itself, which is excluded
                automatically).

        Returns:
            Dict keyed by metric name, each containing:
                ``value``       – the stock's own multiple,
                ``sector_median`` – median of peer multiples,
                ``percentile``  – percentile rank (0-100),
                ``z_score``     – standard deviations from peer mean.
        """
        logger.info("Sector comparison for %s vs %d peers", symbol, len(sector_data))

        # Build peer arrays per metric (exclude the target symbol)
        peer_symbols = [s for s in sector_data if s != symbol]
        if not peer_symbols:
            logger.warning("No sector peers found for %s", symbol)
            return {}

        comparison: Dict[str, Dict[str, Optional[float]]] = {}

        for metric in ("PE", "PB", "EV_EBITDA", "EV_Sales", "PEG"):
            own_val = multiples.get(metric)
            peer_vals = [
                sector_data[s].get(metric)
                for s in peer_symbols
                if sector_data[s].get(metric) is not None
            ]

            if not peer_vals or own_val is None:
                comparison[metric] = {
                    "value": own_val,
                    "sector_median": None,
                    "percentile": None,
                    "z_score": None,
                }
                continue

            arr = np.array(peer_vals, dtype=float)
            median = float(np.median(arr))
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

            # Percentile rank of own_val within the peer distribution
            pct = float(np.sum(arr < own_val) / len(arr) * 100)

            z = round((own_val - mean) / std, 4) if std > 0 else 0.0

            comparison[metric] = {
                "value": own_val,
                "sector_median": round(median, 4),
                "percentile": round(pct, 2),
                "z_score": z,
            }

        logger.debug("Sector comparison result: %s", comparison)
        return comparison

    # --------------------------------------------------------------------- #
    # Fair-value from peers
    # --------------------------------------------------------------------- #

    @staticmethod
    def fair_value_from_peers(
        symbol: str,
        peer_multiples: Dict[str, Dict[str, Optional[float]]],
        target_earnings: float,
        target_book: float,
        target_ebitda: float = 0.0,
        target_revenue: float = 0.0,
    ) -> Dict[str, Any]:
        """Derive a fair-value range by applying peer-median multiples.

        For each metric (PE, PB, EV/EBITDA, EV/Sales), computes an
        implied fair value by multiplying the sector median multiple
        with the target company's corresponding financial metric.

        Args:
            symbol: Ticker being valued.
            peer_multiples: ``{peer: {PE, PB, …}}``.
            target_earnings: EPS of the target stock.
            target_book: Book value per share.
            target_ebitda: EBITDA (total, ₹ Cr).
            target_revenue: Revenue (total, ₹ Cr).

        Returns:
            Dict with ``implied_values`` per metric, ``fair_value_low``,
            ``fair_value_mid``, ``fair_value_high``.
        """
        logger.info("Fair value from peers for %s", symbol)

        peers_only = {s: m for s, m in peer_multiples.items() if s != symbol}
        if not peers_only:
            logger.warning("No peer data available for %s", symbol)
            return {
                "implied_values": {},
                "fair_value_low": None,
                "fair_value_mid": None,
                "fair_value_high": None,
            }

        # Collect peer medians for each metric
        def _peer_median(metric: str) -> Optional[float]:
            vals = [
                m[metric]
                for m in peers_only.values()
                if m.get(metric) is not None
            ]
            return float(np.median(vals)) if vals else None

        implied: Dict[str, Optional[float]] = {}

        pe_median = _peer_median("PE")
        if pe_median is not None and target_earnings > 0:
            implied["PE_implied"] = round(pe_median * target_earnings, 2)

        pb_median = _peer_median("PB")
        if pb_median is not None and target_book > 0:
            implied["PB_implied"] = round(pb_median * target_book, 2)

        ev_ebitda_median = _peer_median("EV_EBITDA")
        if ev_ebitda_median is not None and target_ebitda > 0:
            implied["EV_EBITDA_implied"] = round(ev_ebitda_median * target_ebitda, 2)

        ev_sales_median = _peer_median("EV_Sales")
        if ev_sales_median is not None and target_revenue > 0:
            implied["EV_Sales_implied"] = round(ev_sales_median * target_revenue, 2)

        valid_vals = [v for v in implied.values() if v is not None]
        if not valid_vals:
            return {
                "implied_values": implied,
                "fair_value_low": None,
                "fair_value_mid": None,
                "fair_value_high": None,
            }

        fair_low = round(min(valid_vals), 2)
        fair_high = round(max(valid_vals), 2)
        fair_mid = round(float(np.median(valid_vals)), 2)

        logger.info(
            "Peer fair-value range for %s: ₹%.2f – ₹%.2f (mid ₹%.2f)",
            symbol, fair_low, fair_high, fair_mid,
        )
        return {
            "implied_values": implied,
            "fair_value_low": fair_low,
            "fair_value_mid": fair_mid,
            "fair_value_high": fair_high,
        }

    # --------------------------------------------------------------------- #
    # Full valuation
    # --------------------------------------------------------------------- #

    def full_valuation(
        self,
        fundamentals: Dict[str, Any],
        sector_data: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    ) -> Dict[str, Any]:
        """End-to-end relative valuation.

        Args:
            fundamentals: Fundamental data for the target stock.
                Expected keys: ``symbol``, ``price``, ``eps``,
                ``book_value_per_share``, ``enterprise_value``,
                ``ebitda``, ``revenue``, ``earnings_growth``,
                ``sector``, ``dividend_per_share``.
            sector_data: ``{peer_symbol: {PE, PB, …}}``.  If ``None``,
                only standalone multiples are computed.

        Returns:
            Dict with ``multiples``, ``sector_comparison``,
            ``fair_value_range`` (low/mid/high), ``z_scores``,
            ``sector``.
        """
        symbol = fundamentals.get("symbol", "UNKNOWN")
        sector_raw = fundamentals.get("sector", "OTHER")
        sector = _normalise_sector(sector_raw)
        logger.info("Full relative valuation for %s (sector=%s)", symbol, sector)

        multiples = self.compute_multiples(fundamentals)

        comparison: Dict[str, Any] = {}
        peer_fv: Dict[str, Any] = {
            "fair_value_low": None,
            "fair_value_mid": None,
            "fair_value_high": None,
            "implied_values": {},
        }
        z_scores: Dict[str, Optional[float]] = {}

        if sector_data:
            comparison = self.sector_comparison(symbol, multiples, sector_data)
            z_scores = {
                metric: info.get("z_score")
                for metric, info in comparison.items()
            }

            peer_fv = self.fair_value_from_peers(
                symbol=symbol,
                peer_multiples=sector_data,
                target_earnings=fundamentals.get("eps", 0.0),
                target_book=fundamentals.get("book_value_per_share", 0.0),
                target_ebitda=fundamentals.get("ebitda", 0.0),
                target_revenue=fundamentals.get("revenue", 0.0),
            )

        return {
            "symbol": symbol,
            "sector": sector,
            "multiples": multiples,
            "sector_comparison": comparison,
            "fair_value_range": {
                "low": peer_fv.get("fair_value_low"),
                "mid": peer_fv.get("fair_value_mid"),
                "high": peer_fv.get("fair_value_high"),
            },
            "implied_values": peer_fv.get("implied_values", {}),
            "z_scores": z_scores,
        }
