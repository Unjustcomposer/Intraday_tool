import time
import logging
import threading
from typing import Callable, Any

logger = logging.getLogger(__name__)

class MarginReconciler:
    """
    Background worker that syncs internal state with the live broker margin/positions API.
    Prevents phantom rejections, over-leveraging, and out-of-sync portfolios.
    """
    def __init__(self, fetch_margin_func: Callable[[], float], sync_interval: int = 10):
        """
        Args:
            fetch_margin_func: A callback to the ExecutionEngine/broker to get live margin.
            sync_interval: Interval in seconds to sync.
        """
        self.fetch_margin_func = fetch_margin_func
        self.sync_interval = sync_interval
        self._live_margin = 0.0
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        logger.info(f"MarginReconciler started (interval={self.sync_interval}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("MarginReconciler stopped")

    def _sync_loop(self):
        while self._running:
            try:
                live_margin = self.fetch_margin_func()
                self._live_margin = live_margin
                # Future: Could update a centralized Redis state or emit a signal here
            except Exception as e:
                logger.error(f"Margin reconciliation failed: {e}")
            
            time.sleep(self.sync_interval)

    @property
    def live_margin(self) -> float:
        return self._live_margin
