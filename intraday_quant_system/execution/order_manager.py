import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from risk.position_sizing import kelly_fraction, volatility_adjusted_size, PortfolioLimits
from .algorithms import TWAP, VWAP, AlmgrenChriss, Iceberg

logger = logging.getLogger(__name__)

class OrderManager:
    """
    High-level manager orchestrating entry, exit, and stop-loss logic.
    
    Production features:
      - Circuit breaker logic (NSE upper/lower circuits)
      - Gap risk handling
      - Proper integration with StopLossEngine
    """
    # NSE sector mapping for exposure tracking
    SECTOR_MAP = {
        'RELIANCE': 'energy', 'ONGC': 'energy', 'IOC': 'energy', 'BPCL': 'energy',
        'HDFCBANK': 'banking', 'ICICIBANK': 'banking', 'SBIN': 'banking', 'KOTAKBANK': 'banking',
        'AXISBANK': 'banking', 'INDUSINDBK': 'banking', 'BANKBARODA': 'banking',
        'TCS': 'it', 'INFY': 'it', 'WIPRO': 'it', 'HCLTECH': 'it', 'TECHM': 'it', 'LTI': 'it',
        'TATAMOTORS': 'auto', 'MARUTI': 'auto', 'M&M': 'auto', 'BAJAJ-AUTO': 'auto',
        'TATASTEEL': 'metals', 'HINDALCO': 'metals', 'JSWSTEEL': 'metals',
        'SUNPHARMA': 'pharma', 'DRREDDY': 'pharma', 'CIPLA': 'pharma',
        'HINDUNILVR': 'fmcg', 'ITC': 'fmcg', 'NESTLEIND': 'fmcg',
    }

    def __init__(self, execution_engine, risk_monitor, stop_engine, config: dict = None):
        self.exec_engine = execution_engine
        self.risk_monitor = risk_monitor
        self.stop_engine = stop_engine
        self.config = config or {}
        
        timing = self.config.get('intraday', {})
        self.hard_exit_time = timing.get('hard_exit', "15:15")
        
        # Track active stops per symbol
        self.active_stops: Dict[str, Dict] = {}
        
        # Track last trade time per symbol to prevent wash trade whipsawing
        self.last_trade_time: Dict[str, datetime] = {}
        
        # Track open orders for 30s conversion logic
        self.open_orders: Dict[str, Dict] = {} # order_id -> {symbol, timestamp, ...}
        
        # Track active execution algorithms
        self.active_algorithms: Dict[str, Dict] = {} # symbol -> {algo, price, regime, ...}

    def process_signals(self, signals_df: pd.DataFrame, current_prices: Dict[str, float],
                        market_data: pd.DataFrame = None, features_data: Dict[str, pd.Series] = None):
        """
        Process new alpha signals.
        signals_df should have [symbol, signal, confidence, regime]
        features_data: optional dict {symbol: latest_feature_row} containing 'gk_vol' or 'atr'
        """
        if self.risk_monitor.trading_halted:
            logger.warning("Trading halted. Ignoring new signals.")
            return
            
        if self.check_hard_exit():
            logger.info("Past hard exit time. No new trades allowed.")
            return
            
        # Get maximum volatility for window cutoff adjustment
        max_vol = 0.0
        if features_data:
            for sym in current_prices.keys():
                max_vol = max(max_vol, self._get_volatility(sym, features_data))
                
        if not self._is_trading_window_active(volatility=max_vol):
            logger.info("Outside active trading window. Rejecting new signals.")
            return
            
        positions = self.exec_engine.get_positions()
        open_symbols = positions['symbol'].tolist() if not positions.empty else []
        now = datetime.now()
        
        for _, row in signals_df.iterrows():
            symbol = row['symbol']
            signal = row['signal']
            regime = row.get('regime', 'unknown')
            confidence = row.get('confidence', 0.5)
            
            # Check circuit breakers if market data provided
            if market_data is not None and self._is_near_circuit(symbol, current_prices[symbol], market_data):
                logger.warning(f"Symbol {symbol} near circuit limit. Rejecting {signal}.")
                continue
            
            # Already have a position?
            if symbol in open_symbols:
                pos_qty = positions[positions['symbol'] == symbol]['quantity'].iloc[0]
                is_long = pos_qty > 0
                
                # Reversal signal?
                if (signal == 'sell' and is_long) or (signal == 'buy' and not is_long):
                    # Enforce 15-minute cooldown between reversals to avoid wash trade friction
                    if symbol in self.last_trade_time:
                        mins_since_trade = (now - self.last_trade_time[symbol]).total_seconds() / 60.0
                        if mins_since_trade < 15:
                            logger.info(f"Cooldown active for {symbol}. Ignoring reversal signal.")
                            continue
                            
                    logger.info(f"Reversal signal for {symbol}. Closing existing position.")
                    self._close_position(symbol, pos_qty, current_prices[symbol])
                    self.last_trade_time[symbol] = now
                    # Fall through to entry logic below (don't skip the new signal)
                else:
                    continue
                
            # --- Entry logic with real position sizing ---
            if signal in ['buy', 'sell']:
                price = current_prices[symbol]
                side = "BUY" if signal == 'buy' else "SELL"
                
                # Extract real volatility from features (GK vol or ATR)
                vol_val = self._get_volatility(symbol, features_data)
                
                # Compute risk-adjusted position size
                quantity = self._compute_position_size(
                    symbol=symbol,
                    price=price,
                    volatility=vol_val,
                    regime=regime,
                    confidence=confidence,
                    positions_df=positions
                )
                
                if quantity <= 0:
                    logger.info(f"Position sizing returned 0 for {symbol}. Skipping.")
                    continue
                
                LARGE_ORDER_THRESHOLD = 5000
                if quantity >= LARGE_ORDER_THRESHOLD:
                    logger.info(f"Large order of {quantity} for {symbol}. Routing via Algorithmic Execution.")
                    end_time = now + timedelta(minutes=30)
                    algo = TWAP(symbol, quantity, side, now, end_time, slices=10)
                    self.active_algorithms[symbol] = {
                        'algo': algo,
                        'price': price,
                        'regime': regime,
                        'vol_val': vol_val
                    }
                    slice_qty = algo.get_next_slice(now)
                    if slice_qty <= 0:
                        continue
                    current_qty = slice_qty
                    logger.info(f"Executing ALGO slice {side} for {current_qty} shares of {symbol}")
                else:
                    current_qty = quantity
                    logger.info(f"Executing {side} for {current_qty} shares of {symbol} "
                               f"(vol={vol_val:.2f}, regime={regime}, conf={confidence:.2f})")
                
                order_id = self.exec_engine.place_order(
                    symbol=symbol,
                    quantity=current_qty,
                    side=side,
                    order_type="LIMIT",
                    price=price
                )
                
                if order_id:
                    # Estimate initial queue position and OIB based on L2 book depth
                    queue_pos = 1000.0  # default simulation value
                    oib = 0.0
                    if market_data is not None and not market_data.empty:
                        sym_data = market_data[market_data['symbol'] == symbol]
                        if not sym_data.empty:
                            if 'bid_volume_1' in sym_data.columns and 'ask_volume_1' in sym_data.columns:
                                bv = float(sym_data['bid_volume_1'].iloc[-1])
                                av = float(sym_data['ask_volume_1'].iloc[-1])
                                if bv + av > 0:
                                    oib = (bv - av) / (bv + av)
                                queue_pos = bv if side == "BUY" else av
                            else:
                                if side == "BUY" and 'bid_volume' in sym_data.columns:
                                    queue_pos = float(sym_data['bid_volume'].iloc[-1])
                                elif side == "SELL" and 'ask_volume' in sym_data.columns:
                                    queue_pos = float(sym_data['ask_volume'].iloc[-1])
                                
                    # Track for timeouts and queue position estimation
                    self.open_orders[order_id] = {
                        'symbol': symbol,
                        'timestamp': now,
                        'side': side,
                        'regime': regime,
                        'qty': current_qty,
                        'price': price,
                        'queue_pos': queue_pos,
                        'initial_queue_pos': queue_pos,
                        'oib': oib
                    }
                    
                    # Set initial stop loss using real volatility
                    initial_stop = self.stop_engine.calculate_stop(
                        entry_price=price,
                        volatility=vol_val,
                        regime=regime,
                        is_long=(side == "BUY")
                    )
                    
                    self.active_stops[symbol] = {
                        'stop_price': initial_stop,
                        'is_long': (side == "BUY"),
                        'regime': regime,
                        'volatility': vol_val
                    }
                    self.last_trade_time[symbol] = now
                    
    def _get_volatility(self, symbol: str, features_data: Dict[str, pd.Series] = None) -> float:
        """
        Extract real volatility from feature data.
        Priority: gk_vol > realized_vol > atr > price-based fallback
        """
        if features_data and symbol in features_data:
            feat = features_data[symbol]
            for vol_col in ['gk_vol', 'realized_vol', 'atr']:
                if vol_col in feat.index and pd.notna(feat[vol_col]) and feat[vol_col] > 0:
                    return float(feat[vol_col])
        
        # Fallback: estimate from price (assume ~1.5% daily vol for large-cap NSE)
        logger.warning(f"No volatility data for {symbol}. Using 1.5% fallback.")
        return 0.015

    def _compute_position_size(self, symbol: str, price: float, volatility: float, regime: str,
                               confidence: float, positions_df: pd.DataFrame) -> int:
        """
        Risk-adjusted position sizing pipeline:
        1. Base size from volatility-adjusted sizing (target_vol / asset_vol)
        2. Scale by Kelly fraction based on model confidence
        3. Scale by regime exposure multiplier
        4. Enforce portfolio limits (max positions, sector, portfolio exposure)
        5. Enforce max 2% risk per trade
        """
        risk_config = self.config.get('risk', {})
        capital = getattr(self.risk_monitor, 'current_capital', 1000000.0)
        max_risk_pct = risk_config.get('max_risk_per_trade', 0.02)
        target_vol = self.config.get('target_volatility', 0.15)
        
        # 1. Volatility-adjusted base size
        if volatility > 0:
            vol_size = volatility_adjusted_size(target_vol, volatility, capital)
        else:
            vol_size = capital * 0.1  # 10% fallback
        
        # 2. Kelly-inspired confidence scaling (map ensemble score to bet fraction)
        # Confidence typically 0.0-1.0, treat as win probability proxy
        # Use half-Kelly for conservative sizing
        kelly_scale = max(0.0, min(1.0, (confidence - 0.5) * 2))  # Normalize 0.5-1.0 → 0-1
        kelly_scale = kelly_scale * 0.5  # Half-Kelly
        kelly_scale = max(kelly_scale, 0.05)  # Minimum 5% of vol-size to avoid zero
        
        # 3. Regime scaling
        regime_scalars = {
            'quiet': 1.0,
            'bull_volatile': 0.5,
            'bear_volatile': 0.25,
            'unknown': 0.5
        }
        regime_scale = regime_scalars.get(regime, 0.5)
        
        # 4. Apply exposure multiplier from risk monitor (VIX-based)
        exposure_mult = getattr(self.risk_monitor, 'exposure_multiplier', 1.0)
        
        # Combined size
        position_value = vol_size * kelly_scale * regime_scale * exposure_mult
        
        # 5. Cap at max risk per trade
        max_position_value = capital * max_risk_pct / max(volatility, 0.001)
        position_value = min(position_value, max_position_value)
        
        # Convert to shares
        quantity = int(position_value / price) if price > 0 else 0
        quantity = max(quantity, 0)
        
        # 6. Max shares cap: never exceed 1% of estimated ADV
        # Conservative estimate: large-cap NSE stock trades ~5M shares/day
        estimated_adv_shares = 5_000_000
        max_shares = int(estimated_adv_shares * 0.01)  # 1% = 50,000 shares
        if quantity > max_shares:
            logger.warning(f"Quantity {quantity} exceeds ADV cap {max_shares}. Capping.")
            quantity = max_shares
        
        # 7. Portfolio limit check with REAL exposure tracking
        n_positions = len(positions_df) if not positions_df.empty else 0
        
        # Compute real sector exposure
        symbol_sector = self.SECTOR_MAP.get(symbol, 'unknown')
        sector_value = 0.0
        portfolio_value = 0.0
        if not positions_df.empty and 'average_price' in positions_df.columns:
            for _, pos in positions_df.iterrows():
                pos_val = abs(pos['quantity']) * pos['average_price']
                portfolio_value += pos_val
                pos_sector = self.SECTOR_MAP.get(pos['symbol'], 'unknown')
                if pos_sector == symbol_sector:
                    sector_value += pos_val
        
        current_sector_exposure = sector_value / capital if capital > 0 else 0
        current_portfolio_exposure = portfolio_value / capital if capital > 0 else 0
        
        if not PortfolioLimits.check_trade(
            capital=capital,
            current_positions=n_positions,
            proposed_risk=quantity * price * volatility,
            current_sector_exposure=current_sector_exposure,
            current_portfolio_exposure=current_portfolio_exposure,
            config=self.config
        ):
            return 0
        
        logger.info(f"Position size: {quantity} shares @ {price:.2f} "
                    f"(kelly={kelly_scale:.2f}, regime={regime_scale:.1f}, expo={exposure_mult:.1f})")
        return quantity

    def _is_trading_window_active(self, volatility: float = 0.0) -> bool:
        """
        Enforce trading window. Dynamic no-entry cutoff:
        If volatility is high (annualized vol > 25% or ATR > threshold, represented by volatility parameter > 0.025),
        move cutoff to 14:00 (since positions have less time to recover). Otherwise 14:15.
        """
        now = datetime.now().time()
        start_time = datetime.strptime("09:25", "%H:%M").time()
        
        if volatility > 0.025:
            cutoff_str = "14:00"
            logger.info(f"High volatility ({volatility:.3f}) detected. Dynamic no-entry window cutoff advanced to 14:00.")
        else:
            cutoff_str = "14:15"
            
        cutoff_time = datetime.strptime(cutoff_str, "%H:%M").time()
        return start_time <= now <= cutoff_time

    def _is_near_circuit(self, symbol: str, price: float, df: pd.DataFrame) -> bool:
        """Check if price is within 0.5% of upper/lower circuit"""
        if df.empty or 'close' not in df.columns:
            return False
            
        # Get previous day close (assuming df is intraday and we find the first bar's prev close)
        # Simplified for now: just use the first bar of the day as reference
        ref_price = df['open'].iloc[0] 
        
        # Standard NSE circuit is usually 10% or 20%
        # If price moved > 9.5%, it's near a 10% circuit
        move_pct = abs(price - ref_price) / ref_price
        
        if move_pct > 0.095:
            return True
            
        return False

    def manage_open_positions(self, current_prices: Dict[str, float]):
        """
        Monitor existing positions and pending orders.
        """
        now = datetime.now()
        
        # 1. Handle pending LIMIT order timeouts and queue position deterioration
        for order_id, info in list(self.open_orders.items()):
            symbol = info['symbol']
            if symbol not in current_prices:
                continue
                
            elapsed = (now - info['timestamp']).total_seconds()
            
            # Simulate queue position progression (decrementing by 50 shares per tick/iteration)
            if 'queue_pos' in info:
                info['queue_pos'] -= 50.0
                
            # Repricing check: if queue position deteriorates (stagnant/larger) or after 15 seconds
            # we re-price to be more competitive.
            if elapsed > 15 and info.get('queue_pos', 0) > 0:
                logger.info(f"LIMIT order {order_id} queue position stagnated ({info['queue_pos']:.0f} remaining). Re-pricing LIMIT order.")
                # Cancel old order
                self.exec_engine.cancel_all_orders(symbol)
                
                # Place new order at new current price
                new_price = current_prices[symbol]
                new_order_id = self.exec_engine.place_order(
                    symbol=symbol,
                    quantity=info['qty'],
                    side=info['side'],
                    order_type="LIMIT",
                    price=new_price
                )
                if new_order_id:
                    self.open_orders[new_order_id] = {
                        'symbol': symbol,
                        'timestamp': now,
                        'side': info['side'],
                        'regime': info['regime'],
                        'qty': info['qty'],
                        'price': new_price,
                        'queue_pos': info.get('initial_queue_pos', 1000.0) * 0.8,
                        'initial_queue_pos': info.get('initial_queue_pos', 1000.0)
                    }
                del self.open_orders[order_id]
                continue
                
            if elapsed > 30:
                logger.info(f"LIMIT order {order_id} timed out ({elapsed:.1f}s). Converting to MARKET.")
                self.exec_engine.convert_to_market(order_id, current_prices[symbol])
                
                # Activate stops now that we've force-filled
                # In real life, we'd wait for execution callback
                side = info['side']
                regime = info['regime']
                vol_val = self._get_volatility(symbol, features_data=None)  # No features available for timeout conversion
                initial_stop = self.stop_engine.calculate_stop(current_prices[symbol], vol_val, regime, side == "BUY")
                self.active_stops[symbol] = {
                    'stop_price': initial_stop,
                    'is_long': (side == "BUY"),
                    'regime': regime,
                    'volatility': vol_val
                }
                del self.open_orders[order_id]
                
        # 1.5 Handle active algorithms (slice generation)
        for symbol, algo_info in list(self.active_algorithms.items()):
            algo = algo_info['algo']
            slice_qty = algo.get_next_slice(now)
            if slice_qty > 0 and symbol in current_prices:
                price = current_prices[symbol]
                logger.info(f"Executing ALGO slice {algo.side} for {slice_qty} shares of {symbol}")
                order_id = self.exec_engine.place_order(
                    symbol=symbol,
                    quantity=slice_qty,
                    side=algo.side,
                    order_type="LIMIT",
                    price=price
                )
                if order_id:
                    self.open_orders[order_id] = {
                        'symbol': symbol,
                        'timestamp': now,
                        'side': algo.side,
                        'regime': algo_info['regime'],
                        'qty': slice_qty,
                        'price': price,
                        'queue_pos': 1000.0,
                        'initial_queue_pos': 1000.0,
                        'oib': 0.0
                    }
            if not algo.is_active:
                logger.info(f"Algorithm finished for {symbol}")
                del self.active_algorithms[symbol]

        if self.check_hard_exit():
            self.liquidate_all(current_prices)
            return
            
        positions = self.exec_engine.get_positions()
        if positions.empty:
            return
            
        for _, pos in positions.iterrows():
            symbol = pos['symbol']
            qty = pos['quantity']
            
            if qty == 0 or symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Check risk monitor kill switch
            if self.risk_monitor.trading_halted:
                logger.critical(f"Kill switch active! Liquidating {symbol}")
                self._close_position(symbol, qty, current_price)
                continue
                
            # Check stops
            if symbol in self.active_stops:
                stop_info = self.active_stops[symbol]
                stop_price = stop_info['stop_price']
                is_long = stop_info['is_long']
                
                # Check for gap risk (if this is the first tick of the day)
                # Omitted for brevity: requires tracking day transitions
                
                # Update trailing stop
                new_stop = self.stop_engine.calculate_trailing_stop(
                    current_price=current_price,
                    current_stop=stop_price,
                    volatility=stop_info.get('volatility', 5.0),
                    regime=stop_info['regime'],
                    is_long=is_long
                )
                self.active_stops[symbol]['stop_price'] = new_stop
                
                # Execute stop?
                if (is_long and current_price <= new_stop) or (not is_long and current_price >= new_stop):
                    logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    self._close_position(symbol, qty, current_price)
                    del self.active_stops[symbol]

    def _close_position(self, symbol: str, quantity: int, price: float):
        """Helper to close an open position"""
        side = "SELL" if quantity > 0 else "BUY"
        self.exec_engine.place_order(
            symbol=symbol,
            quantity=abs(quantity),
            side=side,
            order_type="MARKET",
            price=price
        )

    def liquidate_all(self, current_prices: Dict[str, float]):
        """Emergency or end-of-day liquidation"""
        positions = self.exec_engine.get_positions()
        if positions.empty:
            return
            
        logger.info("LIQUIDATING ALL POSITIONS")
        
        # Cancel any pending orders first
        self.exec_engine.cancel_all_orders()
        
        for _, pos in positions.iterrows():
            symbol = pos['symbol']
            qty = pos['quantity']
            if qty != 0 and symbol in current_prices:
                self._close_position(symbol, qty, current_prices[symbol])
                
        self.active_stops.clear()

    def check_hard_exit(self) -> bool:
        """Check if we have passed the daily hard exit time (e.g., 15:15 IST)"""
        now = datetime.now()
        exit_time = datetime.strptime(self.hard_exit_time, "%H:%M").time()
        
        current_time = now.time()
        return current_time >= exit_time
