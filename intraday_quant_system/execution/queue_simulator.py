import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class QueuePositionSimulator:
    """
    Simulates limit order execution accounting for queue position and adverse selection.
    Uses 1-minute OHLCV data to proxy queue depletion, as Level-2 data is not available.
    """

    def __init__(self, queue_ahead_fraction: float = 0.5, timeout_bars: int = 15):
        """
        queue_ahead_fraction: Tunable parameter (0.0 to 1.0) representing the assumption 
        of how much printed volume was resting ahead of us in the queue.
        timeout_bars: Number of bars to wait before forcing a market fill.
        """
        self.queue_ahead_fraction = queue_ahead_fraction
        self.timeout_bars = timeout_bars
        self.active_order = None

    def place_order(self, side: str, price: float, qty: int, bar_volume: float):
        """
        Places a limit order into the simulated queue.
        Initial queue position is estimated as a fraction of the current bar's volume.
        """
        self.active_order = {
            "side": side,
            "price": price,
            "qty": qty,
            # Estimate how many shares are ahead of us at this price level
            "queue_ahead": bar_volume * self.queue_ahead_fraction,
            "filled_qty": 0,
            "bars_in_queue": 0,
        }

    def update(self, bar: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Evaluates the active order against the new 1-minute bar.
        Returns a fill event dictionary if the order is completely filled, else None.
        """
        if not self.active_order:
            return None

        self.active_order["bars_in_queue"] += 1
        side = self.active_order["side"]
        limit_price = self.active_order["price"]

        # 2.1 Perpetual Limit Order Fix: Timeout
        if self.active_order["bars_in_queue"] >= self.timeout_bars:
            # Force a market fill at the CURRENT bar's open price
            # (In reality, we would pay spread, the backtest runner adds slippage later)
            self.active_order["filled_qty"] = self.active_order["qty"]
            fill_event = self.active_order.copy()
            fill_event["fill_price"] = bar["open"]
            fill_event["fill_type"] = "timeout_market"
            self.active_order = None
            return fill_event

        # Check if swept or touched
        is_swept = False
        is_touched = False

        if side == "buy":
            if bar["low"] < limit_price:
                is_swept = True
            elif bar["low"] == limit_price:
                is_touched = True
        else:  # sell
            if bar["high"] > limit_price:
                is_swept = True
            elif bar["high"] == limit_price:
                is_touched = True

        if is_swept:
            # Swept through our limit price. 100% filled.
            # Note: Being swept often means adverse selection (price ran over us).
            self.active_order["filled_qty"] = self.active_order["qty"]
            fill_event = self.active_order.copy()
            fill_event["fill_price"] = limit_price
            fill_event["fill_type"] = "swept"
            self.active_order = None
            return fill_event

        elif is_touched:
            # Touched but not swept. Decrement our queue position by the volume printed.
            # We assume a conservative 10% of the bar's volume traded exactly at the high/low extreme.
            trade_volume_at_touch = bar["volume"] * 0.1

            self.active_order["queue_ahead"] -= trade_volume_at_touch

            if self.active_order["queue_ahead"] <= 0:
                # We reached the front of the queue
                remaining_qty = self.active_order["qty"] - self.active_order["filled_qty"]
                # We can fill up to the negative overflow of queue_ahead
                fillable_qty = abs(self.active_order["queue_ahead"])

                new_fill = min(remaining_qty, fillable_qty)
                self.active_order["filled_qty"] += new_fill

                if self.active_order["filled_qty"] >= self.active_order["qty"]:
                    fill_event = self.active_order.copy()
                    fill_event["fill_price"] = limit_price
                    fill_event["fill_type"] = "touched_and_depleted"
                    self.active_order = None
                    return fill_event

        return None

    def cancel_order(self):
        self.active_order = None
