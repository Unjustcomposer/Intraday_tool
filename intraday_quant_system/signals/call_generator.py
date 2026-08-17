import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeCall:
    """Structured intraday trade call for Indian stocks."""

    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    confidence: float  # Meta-labeler probability as percentage (0-100), derived from meta_prob (0-1)
    regime: str
    vix: float
    atr: float
    timestamp: datetime
    shap_features: str = ""
    news_sentiment: float = 0.0
    valid_until: str = "15:10 IST"
    status: str = "ACTIVE"

    @property
    def risk_pct(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price * 100

    @property
    def reward_pct_t1(self) -> float:
        return abs(self.target_1 - self.entry_price) / self.entry_price * 100

    @property
    def reward_pct_t2(self) -> float:
        return abs(self.target_2 - self.entry_price) / self.entry_price * 100

    @property
    def suggested_allocation_pct(self) -> float:
        """Kelly Criterion allocation using meta-labeler probability as win rate.
        
        Kelly f = p - (1-p)/R where:
        - p = meta-labeler probability (confidence / 100) = P(trade profitable)
        - R = risk_reward ratio (target_1 reward / risk)
        
        Half-Kelly for safety, capped at 10% max allocation per trade.
        """
        if self.risk_reward <= 0:
            return 0.0
        win_rate = self.confidence / 100.0  # Convert from percentage to probability
        kelly_fraction = win_rate - ((1.0 - win_rate) / self.risk_reward)
        # Half-Kelly for safety, capped at 10% max allocation per trade
        safe_kelly = max(0.0, min(0.10, kelly_fraction * 0.5))
        return round(safe_kelly * 100.0, 1)

    def update_trailing_stop(self, current_price: float):
        """Update the stop loss if price moves favorably (e.g. 1 ATR trailing)"""
        if self.direction == "BUY":
            new_stop = current_price - (self.atr * 1.0)
            if new_stop > self.stop_loss:
                self.stop_loss = round(new_stop, 2)
        else:
            new_stop = current_price + (self.atr * 1.0)
            if new_stop < self.stop_loss:
                self.stop_loss = round(new_stop, 2)


class CallGenerator:
    """
    Converts raw ensemble scores into structured trade calls with:
      - Entry, Stop Loss, Target 1, Target 2
      - Risk:Reward filtering (minimum 1:2)
      - NSE market hours enforcement (9:15 - 15:10 IST)
      - Confidence gating via Meta-Labeler
    """

    def __init__(
        self,
        min_risk_reward: float = 2.0,
        min_confidence: float = 0.60,
        no_new_calls_after: str = "14:30",
        max_net_exposure: int = 3,
        max_total_calls: int = 5,
    ):
        self.min_risk_reward = min_risk_reward
        self.min_confidence = min_confidence
        self.max_net_exposure = max_net_exposure
        self.max_total_calls = max_total_calls
        self.no_new_calls_after = datetime.strptime(no_new_calls_after, "%H:%M").time()
        self.calls_today: list[TradeCall] = []

    def generate_call(
        self,
        symbol: str,
        signal: str,
        close: float,
        atr: float,
        confidence: float,
        regime: str,
        vix: float,
        vpvr_support: float = None,
        vpvr_resistance: float = None,
        news_sentiment: float = 0.0,
        shap_features: str = "",
        timestamp: datetime = None,
    ) -> TradeCall | None:
        """
        Generate a structured trade call from a raw signal.

        Args:
            symbol: NSE symbol (e.g., 'RELIANCE.NS')
            signal: 'buy' or 'sell' from ensemble
            close: Current close price
            atr: Current ATR value (for SL/TP computation)
            confidence: Meta-labeler confidence (0-1)
            regime: Current market regime string
            vix: India VIX value
            timestamp: Signal timestamp

        Returns:
            TradeCall if it passes all filters, None otherwise.
        """
        timestamp = timestamp or datetime.now()

        # --- Filter 1: Market hours check ---
        current_time = (
            timestamp.time() if hasattr(timestamp, "time") else datetime.now().time()
        )
        if current_time > self.no_new_calls_after:
            logger.info(
                f"[{symbol}] Signal rejected: past {self.no_new_calls_after} cutoff"
            )
            return None

        # --- Portfolio Exposure Limit Check ---
        buy_calls = sum(1 for c in self.calls_today if c.direction == "BUY")
        sell_calls = sum(1 for c in self.calls_today if c.direction == "SELL")
        net_exposure = buy_calls - sell_calls
        total_calls = buy_calls + sell_calls

        if total_calls >= self.max_total_calls:
            logger.info(
                f"[{symbol}] Signal rejected: Max daily trades reached ({self.max_total_calls})"
            )
            return None

        if signal == "buy" and net_exposure >= self.max_net_exposure:
            logger.info(
                f"[{symbol}] BUY rejected: Max net long portfolio exposure reached (+{self.max_net_exposure})"
            )
            return None

        if signal == "sell" and net_exposure <= -self.max_net_exposure:
            logger.info(
                f"[{symbol}] SELL rejected: Max net short portfolio exposure reached (-{self.max_net_exposure})"
            )
            return None

        # --- NLP Boost / Veto ---
        adjusted_min_confidence = self.min_confidence

        # Veto trades that fight the news
        if signal == "buy" and news_sentiment < -0.3:
            logger.info(
                f"[{symbol}] BUY rejected: negative news sentiment ({news_sentiment})"
            )
            return None
        if signal == "sell" and news_sentiment > 0.3:
            logger.info(
                f"[{symbol}] SELL rejected: positive news sentiment ({news_sentiment})"
            )
            return None

        # Boost trades that follow strong news (lower the confidence requirement by up to 5%)
        if signal == "buy" and news_sentiment > 0.3:
            boost = min(0.05, news_sentiment * 0.1)
            adjusted_min_confidence -= boost
            logger.info(
                f"[{symbol}] BUY news boost! Adjusted threshold from {self.min_confidence} to {adjusted_min_confidence:.4f}"
            )
        elif signal == "sell" and news_sentiment < -0.3:
            boost = min(0.05, abs(news_sentiment) * 0.1)
            adjusted_min_confidence -= boost
            logger.info(
                f"[{symbol}] SELL news boost! Adjusted threshold from {self.min_confidence} to {adjusted_min_confidence:.4f}"
            )

        # --- Filter 2: Confidence gate ---
        if confidence < adjusted_min_confidence:
            logger.info(
                f"[{symbol}] Signal rejected: confidence {confidence:.2f} < {adjusted_min_confidence:.4f} (Required)"
            )
            return None
        # --- Filter 3: Signal must be actionable ---
        if signal not in ("buy", "sell"):
            return None

        # --- Compute levels ---
        if atr <= 0:
            logger.warning(f"[{symbol}] ATR is zero or negative, cannot compute levels")
            return None

        # Minimum stop distance: at least 0.5% of entry price or 0.5*ATR, whichever is larger
        min_stop_distance = max(close * 0.005, 0.5 * atr)
        # Maximum target distance: at most 5% of entry price or 3*ATR, whichever is smaller
        max_target_distance = min(close * 0.05, 3.0 * atr)

        if signal == "buy":
            direction = "BUY"
            # Stop Loss: Place just BELOW structural support
            raw_sl = (vpvr_support - 0.2 * atr) if vpvr_support else close - 1.0 * atr
            if abs(close - raw_sl) < min_stop_distance:
                stop_loss = round(close - 1.0 * atr, 2)
            else:
                stop_loss = round(raw_sl, 2)
            if stop_loss >= close:
                stop_loss = round(close - 1.0 * atr, 2)

            # Target 1: Front-run overhead resistance
            raw_t1 = (
                (vpvr_resistance - 0.1 * atr) if vpvr_resistance else close + 1.5 * atr
            )
            if raw_t1 <= close or abs(raw_t1 - close) > max_target_distance:
                target_1 = round(close + 1.5 * atr, 2)
            else:
                target_1 = round(raw_t1, 2)

            # Target 2: Asymmetric continuation (1 ATR beyond T1)
            target_2 = round(target_1 + 1.0 * atr, 2)
        else:
            direction = "SELL"
            # Stop Loss: Place just ABOVE structural resistance
            raw_sl = (
                (vpvr_resistance + 0.2 * atr) if vpvr_resistance else close + 1.0 * atr
            )
            if abs(raw_sl - close) < min_stop_distance:
                stop_loss = round(close + 1.0 * atr, 2)
            else:
                stop_loss = round(raw_sl, 2)
            if stop_loss <= close:
                stop_loss = round(close + 1.0 * atr, 2)

            # Target 1: Front-run underlying support
            raw_t1 = (vpvr_support + 0.1 * atr) if vpvr_support else close - 1.5 * atr
            if raw_t1 >= close or abs(close - raw_t1) > max_target_distance:
                target_1 = round(close - 1.5 * atr, 2)
            else:
                target_1 = round(raw_t1, 2)

            # Target 2: Asymmetric continuation (1 ATR beyond T1)
            target_2 = round(target_1 - 1.0 * atr, 2)

        # --- Compute Risk:Reward ---
        risk = abs(close - stop_loss)
        reward = abs(target_1 - close)
        risk_reward = reward / risk if risk > 0 else 0

        # --- Filter 4: Risk:Reward minimum ---
        if risk_reward < self.min_risk_reward:
            logger.info(
                f"[{symbol}] Signal rejected: R:R {risk_reward:.2f} < {self.min_risk_reward}"
            )
            return None

        # --- Filter X: Minimum Profit Margin ---
        reward_pct = (reward / close) * 100
        if reward_pct < 0.75:
            logger.info(
                f"[{symbol}] Signal rejected: T1 margin {reward_pct:.2f}% < 0.75% (too thin for taxes/fees)"
            )
            return None

        # --- Filter Y: Maximum Stop Loss (Fix #1) ---
        risk_pct_val = (risk / close) * 100
        if risk_pct_val > 1.5:
            logger.info(
                f"[{symbol}] Signal rejected: Stop Loss {risk_pct_val:.2f}% > 1.5% (risk too wide)"
            )
            return None

        # --- Filter 5: R:R sanity cap (anything > 10:1 is likely data artifact) ---
        if risk_reward > 10.0:
            logger.warning(
                f"[{symbol}] R:R {risk_reward:.1f} capped to 10.0 (likely VPVR artifact)"
            )
            # Recompute with pure ATR levels
            if direction == "BUY":
                stop_loss = round(close - 1.0 * atr, 2)
                target_1 = round(close + 2.0 * atr, 2)
                target_2 = round(close + 3.5 * atr, 2)
            else:
                stop_loss = round(close + 1.0 * atr, 2)
                target_1 = round(close - 2.0 * atr, 2)
                target_2 = round(close - 3.5 * atr, 2)
            risk = abs(close - stop_loss)
            reward = abs(target_1 - close)
            risk_reward = reward / risk if risk > 0 else 0

        # --- Filter 6: VIX extreme filter ---
        if vix > 30:
            logger.info(
                f"[{symbol}] Signal rejected: VIX {vix:.1f} > 30 (extreme volatility)"
            )
            return None

        call = TradeCall(
            symbol=symbol.replace(".NS", ""),
            direction=direction,
            entry_price=round(close, 2),
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            risk_reward=round(risk_reward, 2),
            confidence=round(confidence * 100, 1),
            regime=regime,
            vix=round(vix, 1),
            atr=round(atr, 2),
            timestamp=timestamp,
            shap_features=shap_features,
            news_sentiment=news_sentiment,
        )

        self.calls_today.append(call)
        return call

    def format_call(self, call: TradeCall) -> str:
        """Format a TradeCall into a beautiful console-printable string."""
        arrow = "[BUY]" if call.direction == "BUY" else "[SELL]"

        lines = [
            "",
            f"+{'=' * 58}+",
            f"|  {arrow} {call.direction} CALL -- {call.symbol} (NSE){' ' * (28 - len(call.symbol))}|",
            f"+{'=' * 58}+",
            f"|  Entry:     Rs.{call.entry_price:>10,.2f}                        |",
            f"|  Target 1:  Rs.{call.target_1:>10,.2f}  ({'+' if call.direction == 'BUY' else '-'}{call.reward_pct_t1:.2f}%)              |",
            f"|  Target 2:  Rs.{call.target_2:>10,.2f}  ({'+' if call.direction == 'BUY' else '-'}{call.reward_pct_t2:.2f}%)              |",
            f"|  Stop Loss: Rs.{call.stop_loss:>10,.2f}  (-{call.risk_pct:.2f}%)               |",
            f"+{'=' * 58}+",
            f"|  Risk:Reward  = 1:{call.risk_reward:.1f}                               |",
            f"|  Kelly Sizing = {call.suggested_allocation_pct:.1f}% of Capital                    |",
            f"|  Confidence   = {call.confidence:.0f}% (Meta-Labeler)                 |",
            f"|  Regime       = {call.regime:<20s}                 |",
            f"|  India VIX    = {call.vix:<38.1f}|",
            f"|  News Sent.   = {call.news_sentiment:<38.2f}|",
            f"|  Drivers      = {call.shap_features[:38]:<38s}|",
            f"|  Valid Until  = {call.valid_until}  (Intraday Only)          |",
            f"+{'=' * 58}+",
        ]
        return "\n".join(lines)

    def get_summary(self) -> str:
        """Return a summary of all calls generated today."""
        if not self.calls_today:
            return "\n[i] No trade calls generated today. All signals filtered out.\n"

        buy_calls = [c for c in self.calls_today if c.direction == "BUY"]
        sell_calls = [c for c in self.calls_today if c.direction == "SELL"]
        avg_rr = np.mean([c.risk_reward for c in self.calls_today])
        avg_conf = np.mean([c.confidence for c in self.calls_today])

        summary = [
            "",
            "=" * 60,
            f"  CALL SUMMARY -- {datetime.now().strftime('%d %b %Y')}",
            "=" * 60,
            f"  Total Calls:    {len(self.calls_today)}",
            f"  BUY Calls:      {len(buy_calls)}",
            f"  SELL Calls:     {len(sell_calls)}",
            f"  Avg R:R:        1:{avg_rr:.1f}",
            f"  Avg Confidence: {avg_conf:.0f}%",
            "=" * 60,
        ]
        return "\n".join(summary)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all calls as a DataFrame for CSV logging."""
        if not self.calls_today:
            return pd.DataFrame()

        records = []
        for c in self.calls_today:
            records.append(
                {
                    "timestamp": c.timestamp,
                    "symbol": c.symbol,
                    "direction": c.direction,
                    "entry_price": c.entry_price,
                    "stop_loss": c.stop_loss,
                    "target_1": c.target_1,
                    "target_2": c.target_2,
                    "risk_reward": c.risk_reward,
                    "confidence": c.confidence,
                    "regime": c.regime,
                    "vix": c.vix,
                    "atr": c.atr,
                    "valid_until": c.valid_until,
                }
            )
        return pd.DataFrame(records)
