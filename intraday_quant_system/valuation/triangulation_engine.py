"""
Valuation Triangulation Engine
================================

Combines DCF, Relative Valuation, and DDM outputs into a single
consensus fair-value estimate with:

- **Sector-dynamic weights** (Banking, IT, FMCG, default).
- **Confidence scoring** based on model agreement.
- **Margin-of-safety** calculation against the current market price.
- **Trading signal** generation (undervalued / fairly_valued / overvalued).

Usage::

    triangulator = ValuationTriangulator()
    result = triangulator.triangulate(fundamentals_dict, sector_data)
    print(triangulator.generate_report("INFY", result))
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

from valuation.dcf_model import DCFModel
from valuation.relative_valuation import RelativeValuation, _normalise_sector
from valuation.ddm_model import DDMModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector-specific default weight presets
# ---------------------------------------------------------------------------
_SECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "BANKING": {"dcf": 0.30, "relative": 0.50, "ddm": 0.20},
    "IT":      {"dcf": 0.50, "relative": 0.40, "ddm": 0.10},
    "FMCG":    {"dcf": 0.30, "relative": 0.30, "ddm": 0.40},
}
_DEFAULT_WEIGHTS: Dict[str, float] = {"dcf": 0.40, "relative": 0.40, "ddm": 0.20}

# Margin-of-safety thresholds
_MOS_UNDERVALUED: float = 0.15     # ≥ 15 % upside → undervalued
_MOS_OVERVALUED: float = -0.10     # ≤ −10 % (i.e. 10 % downside) → overvalued


class ValuationTriangulator:
    """Master engine that triangulates DCF, Relative, and DDM valuations.

    Args:
        config: Optional dict to override default model weights, e.g.::

            {
                "weights": {"dcf": 0.5, "relative": 0.3, "ddm": 0.2},
                "mos_undervalued": 0.20,
                "mos_overvalued": -0.10,
            }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._dcf = DCFModel()
        self._relative = RelativeValuation()
        self._ddm = DDMModel()
        logger.info("ValuationTriangulator initialised.")

    # --------------------------------------------------------------------- #
    # Weight computation
    # --------------------------------------------------------------------- #

    def _compute_weights(
        self,
        fundamentals: Dict[str, Any],
    ) -> Dict[str, float]:
        """Determine model weights based on sector and config overrides.

        Priority order:
            1. Explicit ``weights`` key in ``self._config``.
            2. Sector-specific preset from ``_SECTOR_WEIGHTS``.
            3. ``_DEFAULT_WEIGHTS``.

        Args:
            fundamentals: Must contain ``sector`` key.

        Returns:
            Dict with ``dcf``, ``relative``, ``ddm`` weights summing to 1.
        """
        # Config override takes precedence
        if "weights" in self._config:
            w = self._config["weights"]
            total = sum(w.values())
            if total > 0:
                return {k: v / total for k, v in w.items()}

        sector = _normalise_sector(fundamentals.get("sector", "OTHER"))
        weights = _SECTOR_WEIGHTS.get(sector, _DEFAULT_WEIGHTS).copy()
        logger.debug("Model weights for sector %s: %s", sector, weights)
        return weights

    # --------------------------------------------------------------------- #
    # Main triangulation
    # --------------------------------------------------------------------- #

    def triangulate(
        self,
        fundamentals: Dict[str, Any],
        sector_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run all three valuation models and produce a consensus estimate.

        Args:
            fundamentals: Fundamental data dict for the target stock.
                Must include ``symbol``, ``price``, and all keys
                required by the individual models.
            sector_data: Peer multiples for relative valuation.
                ``{peer_symbol: {PE, PB, …}}``.

        Returns:
            Dict with:
                ``dcf_value``             – float
                ``relative_value_range``  – (low, mid, high) or None
                ``ddm_value``             – float or None
                ``consensus_fair_value``  – weighted average
                ``margin_of_safety``      – (consensus − price) / consensus
                ``signal``                – 'undervalued' / 'fairly_valued' / 'overvalued'
                ``confidence``            – float 0–1
                ``weights``               – dict used
                ``model_details``         – per-model raw outputs
        """
        symbol = fundamentals.get("symbol", "UNKNOWN")
        current_price = fundamentals.get("price", 0.0)
        logger.info("Triangulating valuation for %s @ ₹%.2f", symbol, current_price)

        # --- Run individual models ----------------------------------------
        dcf_result = self._run_dcf(fundamentals)
        relative_result = self._run_relative(fundamentals, sector_data)
        ddm_result = self._run_ddm(fundamentals)

        dcf_value = dcf_result.get("fair_value")
        ddm_value = ddm_result.get("fair_value")

        rel_range = relative_result.get("fair_value_range", {})
        rel_low = rel_range.get("low")
        rel_mid = rel_range.get("mid")
        rel_high = rel_range.get("high")
        relative_value_range: Optional[Tuple[float, float, float]] = None
        if rel_low is not None and rel_mid is not None and rel_high is not None:
            relative_value_range = (rel_low, rel_mid, rel_high)

        # --- Compute weights (may drop DDM if unavailable) ----------------
        weights = self._compute_weights(fundamentals)
        values: Dict[str, Optional[float]] = {
            "dcf": dcf_value if _is_valid(dcf_value) else None,
            "relative": rel_mid,
            "ddm": ddm_value,
        }

        # Re-normalise weights to exclude models that returned None
        active_weights = {
            k: weights[k] for k in weights if values.get(k) is not None
        }
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}
        else:
            logger.error("All models returned None for %s", symbol)
            return self._empty_result(symbol, current_price, weights)

        # --- Consensus fair value (weighted average) ----------------------
        consensus = sum(
            active_weights[k] * values[k]  # type: ignore[operator]
            for k in active_weights
        )
        consensus = round(consensus, 2)

        # --- Margin of safety ---------------------------------------------
        if consensus > 0:
            margin_of_safety = round(
                (consensus - current_price) / consensus, 4,
            )
        else:
            margin_of_safety = 0.0

        # --- Signal -------------------------------------------------------
        mos_under = self._config.get("mos_undervalued", _MOS_UNDERVALUED)
        mos_over = self._config.get("mos_overvalued", _MOS_OVERVALUED)

        if margin_of_safety >= mos_under:
            signal = "undervalued"
        elif margin_of_safety <= mos_over:
            signal = "overvalued"
        else:
            signal = "fairly_valued"

        # --- Confidence (model agreement) ---------------------------------
        confidence = self._compute_confidence(values, consensus)

        logger.info(
            "Triangulation for %s: consensus=₹%.2f, MoS=%.2f%%, "
            "signal=%s, confidence=%.2f",
            symbol, consensus, margin_of_safety * 100, signal, confidence,
        )

        return {
            "symbol": symbol,
            "current_price": current_price,
            "dcf_value": dcf_value if _is_valid(dcf_value) else None,
            "relative_value_range": relative_value_range,
            "ddm_value": ddm_value,
            "consensus_fair_value": consensus,
            "margin_of_safety": margin_of_safety,
            "signal": signal,
            "confidence": confidence,
            "weights": active_weights,
            "model_details": {
                "dcf": dcf_result,
                "relative": relative_result,
                "ddm": ddm_result,
            },
        }

    # --------------------------------------------------------------------- #
    # Model runners (isolated so failures are contained)
    # --------------------------------------------------------------------- #

    def _run_dcf(self, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DCF model with error containment."""
        try:
            return self._dcf.full_valuation(fundamentals)
        except Exception as exc:
            logger.exception("DCF model failed: %s", exc)
            return {"fair_value": float("nan"), "error": str(exc)}

    def _run_relative(
        self,
        fundamentals: Dict[str, Any],
        sector_data: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Execute Relative Valuation model with error containment."""
        try:
            return self._relative.full_valuation(fundamentals, sector_data)
        except Exception as exc:
            logger.exception("Relative model failed: %s", exc)
            return {"fair_value_range": {}, "error": str(exc)}

    def _run_ddm(self, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DDM model with error containment."""
        try:
            return self._ddm.full_valuation(fundamentals)
        except Exception as exc:
            logger.exception("DDM model failed: %s", exc)
            return {"fair_value": None, "error": str(exc)}

    # --------------------------------------------------------------------- #
    # Confidence scoring
    # --------------------------------------------------------------------- #

    @staticmethod
    def _compute_confidence(
        values: Dict[str, Optional[float]],
        consensus: float,
    ) -> float:
        """Measure model agreement as a 0–1 confidence score.

        Uses **coefficient of variation** (CV) of the non-null model
        outputs.  Lower CV → higher confidence.  The mapping is:

        - CV = 0   →  confidence = 1.0
        - CV ≥ 0.5 →  confidence = 0.0

        A linear interpolation is used in between.

        Args:
            values: ``{model_name: fair_value}`` (may contain None).
            consensus: The weighted-average consensus value.

        Returns:
            Confidence in [0.0, 1.0].
        """
        valid = [v for v in values.values() if v is not None and not math.isnan(v)]
        if len(valid) <= 1:
            return 0.5  # Single model — moderate confidence by definition

        arr = np.array(valid, dtype=float)
        mean_val = np.mean(arr)
        if mean_val == 0:
            return 0.5

        cv = float(np.std(arr, ddof=1) / abs(mean_val))
        # Linear map: CV 0→1.0, CV 0.5→0.0, clamped [0, 1]
        confidence = max(0.0, min(1.0, 1.0 - cv * 2.0))
        return round(confidence, 4)

    # --------------------------------------------------------------------- #
    # Report generation
    # --------------------------------------------------------------------- #

    @staticmethod
    def generate_report(symbol: str, result: Dict[str, Any]) -> str:
        """Format a human-readable valuation summary.

        Args:
            symbol: Stock ticker.
            result: Output of :meth:`triangulate`.

        Returns:
            Multi-line formatted string.
        """
        sep = "=" * 60
        lines = [
            sep,
            f"  VALUATION REPORT — {symbol}",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            sep,
            "",
            f"  Current Price       : ₹{result.get('current_price', 'N/A'):>12}",
            f"  Consensus Fair Value: ₹{result.get('consensus_fair_value', 'N/A'):>12}",
            f"  Margin of Safety    :  {_fmt_pct(result.get('margin_of_safety'))}",
            f"  Signal              :  {result.get('signal', 'N/A').upper()}",
            f"  Confidence          :  {_fmt_pct(result.get('confidence'))}",
            "",
            "-" * 60,
            "  MODEL BREAKDOWN",
            "-" * 60,
        ]

        # DCF
        dcf_v = result.get("dcf_value")
        lines.append(
            f"  DCF Value           : ₹{dcf_v:>12.2f}" if _is_valid(dcf_v)
            else "  DCF Value           :  N/A"
        )

        # Relative
        rel = result.get("relative_value_range")
        if rel:
            lines.append(
                f"  Relative Value      : ₹{rel[0]:.2f}  –  "
                f"₹{rel[1]:.2f}  –  ₹{rel[2]:.2f}  (low / mid / high)"
            )
        else:
            lines.append("  Relative Value      :  N/A")

        # DDM
        ddm_v = result.get("ddm_value")
        lines.append(
            f"  DDM Value           : ₹{ddm_v:>12.2f}" if ddm_v is not None
            else "  DDM Value           :  N/A (non-dividend payer)"
        )

        # Weights
        w = result.get("weights", {})
        lines.extend([
            "",
            "-" * 60,
            "  MODEL WEIGHTS",
            "-" * 60,
        ])
        for model, weight in w.items():
            lines.append(f"  {model.upper():<20s}: {weight:.1%}")

        lines.append(sep)
        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _empty_result(
        symbol: str,
        price: float,
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """Return a safe empty result when all models fail."""
        return {
            "symbol": symbol,
            "current_price": price,
            "dcf_value": None,
            "relative_value_range": None,
            "ddm_value": None,
            "consensus_fair_value": None,
            "margin_of_safety": None,
            "signal": "insufficient_data",
            "confidence": 0.0,
            "weights": weights,
            "model_details": {},
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _is_valid(value: Any) -> bool:
    """Return True if *value* is a finite number."""
    if value is None:
        return False
    try:
        return not math.isnan(value) and not math.isinf(value)
    except TypeError:
        return False


def _fmt_pct(value: Any) -> str:
    """Format a decimal as a percentage string, e.g. 0.15 → '15.00 %'."""
    if value is None:
        return "N/A"
    try:
        return f"{value * 100:>8.2f} %"
    except (TypeError, ValueError):
        return "N/A"
