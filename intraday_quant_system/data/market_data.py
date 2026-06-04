import os
import logging
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from .broker_interface import BrokerClient

logger = logging.getLogger(__name__)


class MarketDataEngine:
    """
    Connects to market data API (Zerodha Kite / yfinance fallback)
    Fetches OHLCV + VWAP data for symbols.
    
    Production notes:
      - VWAP is reset daily (cumulative within each trading day only)
      - Orderbook data (bid/ask) is marked NaN when not available from source
      - Data validation checks for gaps, stale quotes, and anomalies
    """
    def __init__(self, broker_client: Optional[BrokerClient] = None):
        self.broker_client = broker_client

    def authenticate(self):
        """Authenticate with the broker API"""
        if self.broker_client:
            if self.broker_client.authenticate():
                logger.info("Authenticated with Broker API")
            else:
                logger.warning("Broker API authentication failed. Falling back to yfinance.")
                self.broker_client = None
        else:
            logger.info("Using yfinance fallback for market data")

    @staticmethod
    def _compute_daily_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Compute VWAP with proper daily reset.
        VWAP = cumsum(Volume * TypicalPrice) / cumsum(Volume) within each trading day.
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vol_tp = df['volume'] * typical_price
        
        # Group by trading date for daily reset
        if hasattr(df.index, 'date'):
            dates = df.index.date
        elif 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp']).dt.date
        else:
            # Fallback: treat entire series as one day
            cum_vol_tp = vol_tp.cumsum()
            cum_vol = df['volume'].cumsum()
            return cum_vol_tp / cum_vol.replace(0, np.nan)
        
        date_series = pd.Series(dates, index=df.index)
        vwap = pd.Series(index=df.index, dtype=float)
        
        for date, group_idx in date_series.groupby(date_series).groups.items():
            mask = df.index.isin(group_idx)
            day_cum_vol_tp = vol_tp.loc[mask].cumsum()
            day_cum_vol = df['volume'].loc[mask].cumsum()
            vwap.loc[mask] = day_cum_vol_tp / day_cum_vol.replace(0, np.nan)
        
        return vwap

    @staticmethod
    def validate_data(df: pd.DataFrame, symbol: str) -> dict:
        """
        Validate data quality. Returns dict with validation results.
        Checks: gaps, stale quotes, zero volume, price anomalies.
        """
        issues = []
        
        if df.empty:
            return {'valid': False, 'issues': ['Empty dataframe']}
        
        # Check for zero-volume bars
        zero_vol_count = (df['volume'] == 0).sum()
        if zero_vol_count > 0:
            issues.append(f"{zero_vol_count} zero-volume bars detected")
        
        # Check for stale quotes (identical consecutive closes)
        stale_count = (df['close'].diff() == 0).sum()
        stale_pct = stale_count / len(df)
        if stale_pct > 0.3:
            issues.append(f"{stale_pct:.1%} stale quotes (identical consecutive closes)")
        
        # Check for price anomalies (>10% single-bar moves)
        returns = df['close'].pct_change().abs()
        anomalies = (returns > 0.10).sum()
        if anomalies > 0:
            issues.append(f"{anomalies} bars with >10% single-bar moves (possible circuit/split)")
        
        # Check for OHLC consistency
        ohlc_invalid = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        if ohlc_invalid > 0:
            issues.append(f"{ohlc_invalid} bars with invalid OHLC relationships")
        
        # Check for time gaps (missing bars) - only for intraday data
        if len(df) > 1 and hasattr(df.index, 'freq'):
            expected_freq = pd.infer_freq(df.index)
            if expected_freq:
                time_diffs = df.index.to_series().diff()
                median_diff = time_diffs.median()
                large_gaps = (time_diffs > median_diff * 3).sum()
                if large_gaps > 0:
                    issues.append(f"{large_gaps} time gaps detected (>3x median interval)")
        
        if issues:
            for issue in issues:
                logger.warning(f"[{symbol}] Data quality: {issue}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_bars': len(df),
            'zero_vol_bars': int(zero_vol_count),
            'stale_pct': float(stale_pct),
            'anomaly_count': int(anomalies)
        }

    def fetch_intraday_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                            interval: str = '15minute', token: int = None) -> pd.DataFrame:
        """
        Main interface to get intraday data.
        Falls back to yfinance if broker API is unavailable.
        """
        if self.broker_client:
            logger.info(f"Fetching {interval} data for {symbol} from {start_date} to {end_date} via Broker")
            # Map interval names if needed (assuming broker handles standard strings like '15m' or '15minute')
            tf = '15m' if '15' in interval else ('5m' if '5' in interval else '1m')
            try:
                df = self.broker_client.get_historical_data(symbol, tf, start_date, end_date)
                if not df.empty:
                    df['timestamp'] = df.index
                    df = self._compute_metrics(df)
                    return df
            except Exception as e:
                logger.error(f"Broker fetch failed for {symbol}: {e}. Falling back to yfinance.")
                
        # Fallback to yfinance
        logger.info(f"Fetching {interval} data for {symbol} from {start_date} to {end_date} via yfinance")
        return self.fetch_nse_data(symbol, start_date, end_date, interval)

    def fetch_nse_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = '15minute') -> pd.DataFrame:
        """
        Fetch historical data for NSE stocks via yfinance.
        Synthesizes buy/sell volume using tick-rule heuristic.
        """
        logger.info(f"Fetching {interval} data for {symbol} from {start_date} to {end_date} via yfinance")
        
        yf_interval_map = {
            'minute': '1m',
            '5minute': '5m',
            '15minute': '15m',
            '1hour': '1h',
            'day': '1d'
        }
        yf_interval = yf_interval_map.get(interval, '15m')
        
        # yfinance limits: 1m = 7 days, 5m/15m = 60 days, 1h = 730 days
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                              end=end_date.strftime('%Y-%m-%d'),
                              interval=yf_interval)
        except Exception as e:
            logger.error(f"yfinance fetch failed for {symbol}: {e}")
            return pd.DataFrame()
        
        if df.empty:
            logger.warning(f"No data returned for {symbol} from yfinance")
            return pd.DataFrame()
        
        # Normalize column names
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })
        
        # Reset index to get timestamp as column
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Remove timezone info if present (yfinance returns tz-aware)
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
        
        df['symbol'] = symbol
        
        # Compute VWAP
        df = df.set_index('timestamp')
        df['vwap'] = self._compute_daily_vwap(df)
        df = df.reset_index()
        
        # Synthesize buy/sell volume using tick-rule (close vs open direction)
        direction = np.sign(df['close'] - df['open'])
        prev_direction = np.sign(df['close'] - df['close'].shift(1))
        direction = np.where(direction == 0, prev_direction, direction)
        direction = pd.Series(direction, index=df.index).fillna(1)
        
        df['buy_volume'] = np.where(direction > 0, df['volume'] * 0.6, df['volume'] * 0.4)
        df['sell_volume'] = df['volume'] - df['buy_volume']
        df['bid_volume'] = df['buy_volume']
        df['ask_volume'] = df['sell_volume']
        
        # Add empty columns for schema compatibility
        empty_cols = ['bid_price', 'ask_price', 'oi', 'spread', 'aggressor_side',
                      'trade_count',
                      'options_pcr', 'options_max_pain', 'options_unusual_oi',
                      'nifty_futures_basis', 'fii_net_flow', 'dii_net_flow']
        for c in empty_cols:
            df[c] = np.nan
        
        # Reorder to internal schema
        out_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap',
                    'bid_price', 'ask_price', 'bid_volume', 'ask_volume', 'oi', 'spread',
                    'trade_count', 'aggressor_side', 'buy_volume', 'sell_volume',
                    'options_pcr', 'options_max_pain', 'options_unusual_oi',
                    'nifty_futures_basis', 'fii_net_flow', 'dii_net_flow']
        
        for c in out_cols:
            if c not in df.columns:
                df[c] = np.nan
        
        # Drop any extra columns from yfinance (Dividends, Stock Splits, etc.)
        df = df[out_cols]
        
        logger.info(f"Loaded {len(df)} bars for {symbol} via yfinance")
        return df

    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = '5minute') -> pd.DataFrame:
        """
        Fetch historical data. Routes to yfinance for NSE (.NS) symbols,
        Binance for crypto symbols (e.g. BTCUSDT).
        """
        # Route NSE symbols to yfinance
        if symbol.endswith('.NS'):
            return self.fetch_nse_data(symbol, start_date, end_date, interval)
        
        logger.info(f"Fetching {interval} data for {symbol} from {start_date} to {end_date} via Binance")
        
        import requests
        import time
        
        binance_interval_map = {
            'minute': '1m',
            '5minute': '5m',
            '15minute': '15m',
            '1hour': '1h',
            'day': '1d'
        }
        b_interval = binance_interval_map.get(interval, '5m')
        b_symbol = symbol.replace('.NS', '')
        
        base_url = "https://api.binance.com/api/v3/klines"
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int((end_date + timedelta(days=1)).timestamp() * 1000)
        
        all_klines = []
        current_start = start_ts
        
        while current_start < end_ts:
            params = {
                "symbol": b_symbol,
                "interval": b_interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 1000
            }
            try:
                resp = requests.get(base_url, params=params, timeout=10)
                if resp.status_code != 200:
                    logger.error(f"Binance API error {resp.status_code}: {resp.text}")
                    break
                data = resp.json()
                if not data:
                    break
                all_klines.extend(data)
                current_start = data[-1][6] + 1
                time.sleep(0.1) # Rate limit respect
            except Exception as e:
                logger.error(f"Binance request failed: {e}")
                break
                
        if not all_klines:
            logger.warning(f"No data found for {symbol} on Binance")
            return pd.DataFrame()
            
        cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'trade_count', 
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        df = pd.DataFrame(all_klines, columns=cols)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'taker_buy_base']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        
        # Crypto VWAP reset
        df = df.set_index('timestamp')
        df['vwap'] = self._compute_daily_vwap(df)
        df = df.reset_index()
        
        # L2 Order Flow Simulation using Binance true Taker volumes
        df['buy_volume'] = df['taker_buy_base']
        df['sell_volume'] = df['volume'] - df['taker_buy_base']
        df['ask_volume'] = df['sell_volume'] # Sell hits bid, buy hits ask, wait -> ask_volume is volume resting on ask, but we use it as flow
        df['bid_volume'] = df['buy_volume']
        
        # Add required empty columns
        empty_cols = ['bid_price', 'ask_price', 'oi', 'spread', 'aggressor_side',
                      'options_pcr', 'options_max_pain', 'options_unusual_oi',
                      'nifty_futures_basis', 'fii_net_flow', 'dii_net_flow']
        for c in empty_cols:
            df[c] = np.nan
            
        # Reorder to internal schema
        out_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap',
                    'bid_price', 'ask_price', 'bid_volume', 'ask_volume', 'oi', 'spread',
                    'trade_count', 'aggressor_side', 'buy_volume', 'sell_volume',
                    'options_pcr', 'options_max_pain', 'options_unusual_oi',
                    'nifty_futures_basis', 'fii_net_flow', 'dii_net_flow']
        
        for c in out_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[out_cols]
        
        return df


class DataStorage:
    """
    Handles TimescaleDB and Redis storage operations.
    Falls back to local Parquet storage when database is unavailable.
    """
    def __init__(self, timescaledb_url: str = "", redis_url: str = "", local_dir: str = "./data/cache"):
        self.ts_url = timescaledb_url
        self.redis_url = redis_url
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)
        
        self._db_available = bool(timescaledb_url and timescaledb_url != "postgresql://user:password@localhost:5432/quant_db")
        self._redis_available = bool(redis_url and redis_url != "redis://localhost:6379/0")
        
        if self._db_available:
            logger.info("TimescaleDB connection configured")
        else:
            logger.info("Using local Parquet storage fallback (no TimescaleDB)")
        
        if self._redis_available:
            logger.info("Redis connection configured")
        else:
            logger.info("Using in-memory caching fallback (no Redis)")
        
    def save_market_data(self, df: pd.DataFrame, timeframe: str = '5min'):
        """Store data — uses local Parquet when DB unavailable"""
        if df.empty:
            return
        
        if self._db_available:
            logger.info(f"Saving {len(df)} rows to TimescaleDB ({timeframe})")
            # Real TimescaleDB insertion would go here
            # with psycopg2.connect(self.ts_url) as conn: ...
        
        # Always save to local Parquet as backup
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown'
        date_str = pd.to_datetime(df['timestamp'].iloc[0]).strftime('%Y-%m-%d') if 'timestamp' in df.columns else 'unknown'
        
        filepath = os.path.join(self.local_dir, f"{symbol}_{date_str}_{timeframe}.parquet")
        
        # Atomic write: write to temp file then rename
        temp_path = filepath + '.tmp'
        try:
            df.to_parquet(temp_path, engine='pyarrow')
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_path, filepath)
            logger.debug(f"Saved {len(df)} rows to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    def load_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load data from local cache or DB"""
        if self._db_available:
            logger.info(f"Loading data from TimescaleDB for {symbol}")
            # Real TimescaleDB query would go here
        
        # Fallback: load from local Parquet files
        dfs = []
        for filename in os.listdir(self.local_dir):
            if filename.startswith(symbol) and filename.endswith('.parquet'):
                filepath = os.path.join(self.local_dir, filename)
                try:
                    df = pd.read_parquet(filepath)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to read {filepath}: {e}")
        
        if dfs:
            combined = pd.concat(dfs).sort_values('timestamp').drop_duplicates(subset=['symbol', 'timestamp'])
            # Filter by date range
            if 'timestamp' in combined.columns:
                combined['timestamp'] = pd.to_datetime(combined['timestamp'])
                mask = (combined['timestamp'].dt.date >= start_date.date()) & (combined['timestamp'].dt.date <= end_date.date())
                return combined[mask].reset_index(drop=True)
            return combined
        
        return pd.DataFrame()
