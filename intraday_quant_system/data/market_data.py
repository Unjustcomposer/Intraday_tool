import os
import logging
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
# In a real environment, we would use:
# from kiteconnect import KiteConnect

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
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None  # KiteConnect(api_key=self.api_key) if api_key else None
        self._instruments_cache = None

    def get_instrument_tokens(self, symbols: List[str], exchange: str = 'NSE') -> dict:
        """
        Fetch instrument tokens for a list of symbols dynamically.
        """
        if not self.kite:
            # Mock mapping for simulation if no API key
            mock_map = {
                "RELIANCE": 738561,
                "HDFCBANK": 341249,
                "TCS": 2953217,
                "INFY": 408065,
                "ICICIBANK": 1270529,
                "SBIN": 779521,
                "TATAMOTORS": 884737,
                "WIPRO": 969473
            }
            return {sym: mock_map.get(sym) for sym in symbols if sym in mock_map}
            
        try:
            if self._instruments_cache is None:
                self._instruments_cache = pd.DataFrame(self.kite.instruments(exchange))
            
            df = self._instruments_cache
            subset = df[df['tradingsymbol'].isin(symbols)]
            return dict(zip(subset['tradingsymbol'], subset['instrument_token']))
        except Exception as e:
            logger.error(f"Failed to fetch instrument tokens: {e}")
            return {}
        
    def authenticate(self, request_token: str = ""):
        """Authenticate with the Kite API"""
        if self.kite:
            # data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            # self.kite.set_access_token(data["access_token"])
            logger.info("Authenticated with Kite API")
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

    def fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str = '5minute') -> pd.DataFrame:
        """
        Fetch historical data.
        Returns DataFrame with columns:
        [symbol, open, high, low, close, volume, vwap]
        
        Orderbook columns (bid/ask) are set to NaN when not available from the data source.
        They will be populated when using Kite API with Level 2 data.
        """
        logger.info(f"Fetching {interval} data for {symbol} from {start_date} to {end_date}")
        
        # Mapping for yfinance
        yf_interval_map = {
            'minute': '1m',
            '5minute': '5m',
            '15minute': '15m',
            'day': '1d'
        }
        yf_interval = yf_interval_map.get(interval, '5m')
        
        # yfinance expects .NS for NSE
        yf_symbol = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
        
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                interval=yf_interval
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date/Datetime a column
            df = df.reset_index()
            if 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'timestamp'})
            elif 'Date' in df.columns:
                df = df.rename(columns={'Date': 'timestamp'})
            
            df.columns = [c.lower() for c in df.columns]
            df['symbol'] = symbol
            
            # Validate required columns
            required_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column {col}")
            
            # Remove dividend and stock split columns if present (yfinance artifacts)
            for drop_col in ['dividends', 'stock splits', 'capital gains']:
                if drop_col in df.columns:
                    df = df.drop(columns=[drop_col])
            
            # Compute proper daily-reset VWAP
            df = df.set_index('timestamp')
            df['vwap'] = self._compute_daily_vwap(df)
            df = df.reset_index()
            
            # Orderbook columns — NaN when not available from data source
            # These will be populated when using Kite API with Level 2 market depth
            orderbook_cols = ['bid_price', 'ask_price', 'bid_volume', 'ask_volume',
                              'oi', 'spread', 'trade_count', 'aggressor_side']
            for col in orderbook_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Compute spread from bid/ask if both available
            bid_ask_available = df['bid_price'].notna() & df['ask_price'].notna()
            if bid_ask_available.any():
                df.loc[bid_ask_available, 'spread'] = (
                    (df.loc[bid_ask_available, 'ask_price'] - df.loc[bid_ask_available, 'bid_price'])
                    / df.loc[bid_ask_available, 'close']
                )
            
            # Add Options Flow and Cross-Asset columns (Simulated/Moked for paper trading)
            np.random.seed(42)  # For consistent mock behavior
            n_rows = len(df)
            
            # 1. Put/Call Ratio (PCR): mean-reverting around 1.0
            pcr = 1.0 + np.sin(np.linspace(0, 10, n_rows)) * 0.2 + np.random.normal(0, 0.05, n_rows)
            df['options_pcr'] = pcr
            
            # 2. Max Pain Strike: stock price rounded to nearest 50 strike increment
            df['options_max_pain'] = np.round(df['close'] / 50.0) * 50.0
            
            # 3. Unusual Options OI Buildup: 0 or 1 indicator
            df['options_unusual_oi'] = np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05])
            
            # 4. Nifty Futures Basis: +0.1% premium with minor volatility
            df['nifty_futures_basis'] = df['close'] * 0.001 + np.random.normal(0, df['close'] * 0.0002, n_rows)
            
            # 5. FII / DII Daily Net Flow (Cr INR): constant per calendar day
            if 'timestamp' in df.columns:
                dates = pd.to_datetime(df['timestamp']).dt.date
            else:
                dates = pd.Series(datetime.now().date(), index=df.index)
            
            unique_dates = np.unique(dates)
            fii_flows = dict(zip(unique_dates, np.random.uniform(-1500, 1500, len(unique_dates))))
            dii_flows = dict(zip(unique_dates, np.random.uniform(-1000, 1000, len(unique_dates))))
            
            df['fii_net_flow'] = [fii_flows[d] for d in dates]
            df['dii_net_flow'] = [dii_flows[d] for d in dates]
            
            # Reorder columns
            cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap',
                    'bid_price', 'ask_price', 'bid_volume', 'ask_volume', 'oi', 'spread',
                    'trade_count', 'aggressor_side',
                    'options_pcr', 'options_max_pain', 'options_unusual_oi',
                    'nifty_futures_basis', 'fii_net_flow', 'dii_net_flow']
            
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
                    
            df = df[cols]
            
            # Run data validation
            validation = self.validate_data(df, symbol)
            if not validation['valid']:
                logger.warning(f"Data quality issues for {symbol}: {validation['issues']}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()


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
            combined = pd.concat(dfs).sort_values('timestamp').drop_duplicates()
            # Filter by date range
            if 'timestamp' in combined.columns:
                combined['timestamp'] = pd.to_datetime(combined['timestamp'])
                mask = (combined['timestamp'].dt.date >= start_date.date()) & (combined['timestamp'].dt.date <= end_date.date())
                return combined[mask].reset_index(drop=True)
            return combined
        
        return pd.DataFrame()
