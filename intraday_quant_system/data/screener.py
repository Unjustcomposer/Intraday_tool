import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Full NSE F&O Universe watchlist (deduplicated) — used as fallback
FNO_UNIVERSE = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "TCS.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "TATAMOTORS.NS",
    "WIPRO.NS",
    "BAJFINANCE.NS",
    "LT.NS",
    "HCLTECH.NS",
    "BHARTIARTL.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "ADANIENT.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "JSWSTEEL.NS",
    "TATASTEEL.NS",
    "TECHM.NS",
    "INDUSINDBK.NS",
    "HINDALCO.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "DIVISLAB.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "M&M.NS",
    "NESTLEIND.NS",
    "SBILIFE.NS",
    "TATACHEM.NS",
    "TATACONSUM.NS",
    "UPL.NS",
    "AMBUJACEM.NS",
    "ASHOKLEY.NS",
    "AUROPHARMA.NS",
    "BANDHANBNK.NS",
    "BANKBARODA.NS",
    "BHEL.NS",
    "BIOCON.NS",
    "BOSCHLTD.NS",
    "CANBK.NS",
    "CHOLAFIN.NS",
    "COFORGE.NS",
    "CONCOR.NS",
    "COROMANDEL.NS",
    "CROMPTON.NS",
    "CUMMINSIND.NS",
    "DABUR.NS",
    "DALBHARAT.NS",
    "DEEPAKNTR.NS",
    "DELTACORP.NS",
    "DIXON.NS",
    "ESCORTS.NS",
    "EXIDEIND.NS",
    "FEDERALBNK.NS",
    "GAIL.NS",
    "GLENMARK.NS",
    "GMRINFRA.NS",
    "GODREJCP.NS",
    "GODREJPROP.NS",
    "GRANULES.NS",
    "GUJGASLTD.NS",
    "HAL.NS",
    "HAVELLS.NS",
    "HDFCAMC.NS",
    "ICICIGI.NS",
    "ICICIPRULI.NS",
    "IDEA.NS",
    "IDFCFIRSTB.NS",
    "IEX.NS",
    "IGL.NS",
    "INDHOTEL.NS",
    "INDIACEM.NS",
    "INDIAMART.NS",
    "INDIGO.NS",
    "INDUSTOWER.NS",
    "IPCALAB.NS",
    "IRCTC.NS",
    "JINDALSTEL.NS",
    "JUBLFOOD.NS",
    "L&TFH.NS",
    "LALPATHLAB.NS",
    "LAURUSLABS.NS",
    "LICHSGFIN.NS",
    "LTIM.NS",
    "LTTS.NS",
    "M&MFIN.NS",
    "MANAPPURAM.NS",
    "MARICO.NS",
    "MGL.NS",
    "MOTHERSON.NS",
    "MPHASIS.NS",
    "MRF.NS",
    "MUTHOOTFIN.NS",
    "NATIONALUM.NS",
    "NAUKRI.NS",
    "NAVINFLUOR.NS",
    "OBEROIRLTY.NS",
    "OFSS.NS",
    "PAGEIND.NS",
    "PEL.NS",
    "PERSISTENT.NS",
    "PETRONET.NS",
    "PFC.NS",
    "PIDILITIND.NS",
    "PIIND.NS",
    "PNB.NS",
    "POLYCAB.NS",
    "PVRINOX.NS",
    "RAMCOCEM.NS",
    "RBLBANK.NS",
    "RECLTD.NS",
    "SAIL.NS",
    "SBICARD.NS",
    "SHREECEM.NS",
    "SIEMENS.NS",
    "SRF.NS",
    "SUNTV.NS",
    "SYNGENE.NS",
    "TATACOMM.NS",
    "TATAPOWER.NS",
    "TORNTPHARM.NS",
    "TRENT.NS",
    "TVSMOTOR.NS",
    "UBL.NS",
    "VEDL.NS",
    "VOLTAS.NS",
    "ZEEL.NS",
    "ZYDUSLIFE.NS",
]


def fetch_all_nse_symbols() -> list[str]:
    """
    Dynamically fetches ALL equity symbols listed on NSE from the official
    NSE India public CSV endpoint.

    Returns:
        List of NSE symbols in yfinance format (e.g., 'RELIANCE.NS')
    """
    import requests

    NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv",
    }

    try:
        logger.info("Fetching full NSE equity list from NSE India archives...")
        resp = requests.get(NSE_EQUITY_URL, headers=headers, timeout=15)
        resp.raise_for_status()

        from io import StringIO

        df = pd.read_csv(StringIO(resp.text))

        # The CSV has a 'SYMBOL' column with the NSE ticker
        if "SYMBOL" in df.columns:
            symbols = [f"{s.strip()}.NS" for s in df["SYMBOL"].dropna().unique()]
            logger.info(f"Fetched {len(symbols)} NSE-listed equities from NSE India.")
            return symbols
        else:
            logger.warning(
                f"NSE CSV did not contain 'SYMBOL' column. Columns: {list(df.columns)}"
            )
            return []

    except Exception as e:
        logger.warning(f"Failed to fetch NSE equity list: {e}. Will try fallback.")
        return []


def fetch_nse_symbols_fallback() -> list[str]:
    """
    Fallback: Fetch NSE symbols from the yfinance-compatible Wikipedia NIFTY 500 list.
    """
    try:
        logger.info("Fetching NIFTY 500 constituents from Wikipedia as fallback...")
        tables = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_500", match="Symbol")
        if tables:
            df = tables[0]
            sym_col = [
                c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()
            ]
            if sym_col:
                symbols = [f"{s.strip()}.NS" for s in df[sym_col[0]].dropna().unique()]
                logger.info(f"Fetched {len(symbols)} NIFTY 500 symbols from Wikipedia.")
                return symbols
    except Exception as e:
        logger.warning(f"Wikipedia fallback also failed: {e}")
    return []


class DynamicScreener:
    """
    Full-Market Dynamic Pre-Market Screener.

    Scans ALL NSE-listed equities (or as many as available) at market open
    to identify "Stocks in Play" based on pre-market gap momentum.
    Uses batched, threaded yfinance downloads to minimize latency.
    """

    def __init__(self, top_n: int = 20):
        self.top_n = top_n
        self.universe = self._build_universe()

    def _build_universe(self) -> list[str]:
        """Build the scanning universe: try full NSE, then NIFTY 500, then F&O fallback."""
        symbols = fetch_all_nse_symbols()
        if len(symbols) >= 500:
            return symbols

        symbols = fetch_nse_symbols_fallback()
        if len(symbols) >= 100:
            return symbols

        logger.warning("Using hardcoded F&O universe as final fallback.")
        return FNO_UNIVERSE

    def _download_batch(
        self, symbols: list[str], batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Download daily data in batches to avoid overwhelming yfinance.
        Returns concatenated DataFrame.
        """
        all_closes = []
        all_opens = []
        all_volumes = []

        total_batches = (len(symbols) + batch_size - 1) // batch_size
        logger.info(f"Downloading data in {total_batches} batches of {batch_size}...")

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(f"  Batch {batch_num}/{total_batches}: {len(batch)} symbols...")

            try:
                df = yf.download(
                    batch, period="5d", interval="1d", threads=True, progress=False
                )

                if df.empty:
                    continue

                if len(batch) == 1:
                    # Single ticker: yfinance returns flat columns
                    ticker = batch[0]
                    if "Close" in df.columns and "Open" in df.columns:
                        closes = df[["Close"]].rename(columns={"Close": ticker})
                        opens = df[["Open"]].rename(columns={"Open": ticker})
                        volumes = (
                            df[["Volume"]].rename(columns={"Volume": ticker})
                            if "Volume" in df.columns
                            else None
                        )
                        all_closes.append(closes)
                        all_opens.append(opens)
                        if volumes is not None:
                            all_volumes.append(volumes)
                else:
                    # Multi ticker: MultiIndex columns
                    if "Close" in df.columns and "Open" in df.columns:
                        all_closes.append(df["Close"])
                        all_opens.append(df["Open"])
                        if "Volume" in df.columns:
                            all_volumes.append(df["Volume"])

            except Exception as e:
                logger.warning(f"  Batch {batch_num} failed: {e}")
                continue

            # Small delay between batches to be polite to Yahoo Finance
            if batch_num < total_batches:
                time.sleep(0.5)

        if not all_closes:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        closes = pd.concat(all_closes, axis=1)
        opens = pd.concat(all_opens, axis=1)
        volumes = pd.concat(all_volumes, axis=1) if all_volumes else pd.DataFrame()

        return closes, opens, volumes

    def scan_pre_market(self) -> list[str]:
        """
        Hybrid screener: combines gap stocks from full NSE + F&O momentum stocks.

        Strategy:
          1. Scan ALL NSE stocks for quality gap stocks (price >= ₹50, volume >= 100K)
          2. Also scan F&O universe for highest recent momentum
          3. Merge both pools, deduplicate, pick top-N

        This ensures we always have quality, liquid stocks to trade.
        """
        logger.info(f"Running FULL MARKET screener on {len(self.universe)} stocks...")

        try:
            # Fix #6: Fetch NIFTY50 baseline for Relative Strength calculation
            nifty_momentum = 0.0
            try:
                nifty = yf.download("^NSEI", period="5d", interval="1d", progress=False)
                if not nifty.empty and "Close" in nifty.columns:
                    # yfinance > 0.2.0 returns DataFrame for single columns
                    if isinstance(nifty["Close"], pd.DataFrame):
                        nifty_closes = nifty["Close"].iloc[:, 0]
                    else:
                        nifty_closes = nifty["Close"]
                    nifty_first = nifty_closes.iloc[0]
                    nifty_last = nifty_closes.iloc[-1]
                    nifty_momentum = (nifty_last - nifty_first) / nifty_first
                    logger.info(
                        f"NIFTY50 5-day baseline momentum: {nifty_momentum*100:+.2f}%"
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch NIFTY50 baseline: {e}")

            closes, opens, volumes = self._download_batch(self.universe)

            if closes.empty or opens.empty:
                logger.error(
                    "Screener failed to download data. Falling back to F&O top-N."
                )
                return FNO_UNIVERSE[: self.top_n]

            # Drop columns with all NaNs (delisted or not found)
            closes = closes.dropna(axis=1, how="all")
            opens = opens.dropna(axis=1, how="all")

            # --- Pool 1: Quality gap stocks from full universe ---
            gap_stocks = []
            rejected_price = 0
            rejected_volume = 0

            for ticker in closes.columns:
                try:
                    ticker_closes = closes[ticker].dropna()
                    ticker_opens = opens[ticker].dropna()

                    if len(ticker_closes) >= 2 and len(ticker_opens) >= 1:
                        prev_close = ticker_closes.iloc[-2]
                        curr_open = ticker_opens.iloc[-1]
                        last_close = ticker_closes.iloc[-1]

                        if last_close < 50:
                            rejected_price += 1
                            continue

                        if prev_close > 0:
                            gap_pct = (curr_open - prev_close) / prev_close

                            avg_vol = 0
                            if not volumes.empty and ticker in volumes.columns:
                                vol_series = volumes[ticker].dropna()
                                if len(vol_series) >= 2:
                                    avg_vol = vol_series.iloc[-3:].mean()

                            avg_turnover = avg_vol * last_close
                            min_turnover = 50_00_000  # ₹50 Lakhs

                            if avg_vol >= 100_000 and avg_turnover >= min_turnover:
                                gap_stocks.append(
                                    (
                                        ticker,
                                        abs(gap_pct),
                                        gap_pct,
                                        avg_vol,
                                        last_close,
                                        "gap",
                                    )
                                )
                            elif ticker in FNO_UNIVERSE:
                                gap_stocks.append(
                                    (
                                        ticker,
                                        abs(gap_pct),
                                        gap_pct,
                                        avg_vol,
                                        last_close,
                                        "fno_gap",
                                    )
                                )
                            else:
                                rejected_volume += 1

                except Exception:
                    continue

            logger.info(
                f"Gap scan: {rejected_price} rejected (price < ₹50), "
                f"{rejected_volume} rejected (low volume), {len(gap_stocks)} quality gap stocks"
            )

            # --- Pool 2: F&O stocks by recent momentum (absolute 5-day return) ---
            fno_momentum = []
            gap_tickers = {t[0] for t in gap_stocks}

            for ticker in FNO_UNIVERSE:
                if ticker in gap_tickers:
                    continue  # Already in gap pool
                try:
                    if ticker in closes.columns:
                        tc = closes[ticker].dropna()
                        if len(tc) >= 2:
                            last_close = tc.iloc[-1]
                            first_close = tc.iloc[0]
                            momentum = (last_close - first_close) / first_close

                            # Fix #6: Relative Strength (RS) against NIFTY50
                            relative_strength = abs(momentum - nifty_momentum)

                            avg_vol = 0
                            if not volumes.empty and ticker in volumes.columns:
                                vol_series = volumes[ticker].dropna()
                                if len(vol_series) >= 2:
                                    avg_vol = vol_series.iloc[-3:].mean()

                            fno_momentum.append(
                                (
                                    ticker,
                                    relative_strength,
                                    0.0,
                                    avg_vol,
                                    last_close,
                                    "fno_momentum",
                                )
                            )
                except Exception:
                    continue

            logger.info(
                f"F&O momentum scan: {len(fno_momentum)} additional F&O stocks by recent momentum"
            )

            # Fix #8: Sector Grouping filter
            FNO_SECTORS = {
                "RELIANCE.NS": "Energy",
                "ONGC.NS": "Energy",
                "BPCL.NS": "Energy",
                "PETRONET.NS": "Energy",
                "HDFCBANK.NS": "Bank",
                "ICICIBANK.NS": "Bank",
                "SBIN.NS": "Bank",
                "AXISBANK.NS": "Bank",
                "KOTAKBANK.NS": "Bank",
                "TCS.NS": "IT",
                "INFY.NS": "IT",
                "WIPRO.NS": "IT",
                "HCLTECH.NS": "IT",
                "TECHM.NS": "IT",
                "SUNPHARMA.NS": "Pharma",
                "DRREDDY.NS": "Pharma",
                "CIPLA.NS": "Pharma",
                "DIVISLAB.NS": "Pharma",
                "TATAMOTORS.NS": "Auto",
                "MARUTI.NS": "Auto",
                "M&M.NS": "Auto",
                "BAJAJ-AUTO.NS": "Auto",
                "JSWSTEEL.NS": "Metal",
                "TATASTEEL.NS": "Metal",
                "HINDALCO.NS": "Metal",
                "ULTRACEMCO.NS": "Cement",
                "GRASIM.NS": "Cement",
                "BAJFINANCE.NS": "Fin",
                "CHOLAFIN.NS": "Fin",
                "ITC.NS": "FMCG",
                "HINDUNILVR.NS": "FMCG",
                "NESTLEIND.NS": "FMCG",
            }

            selected = []
            sector_counts = {}
            max_per_sector = max(2, self.top_n // 4)  # Max 2 or 25% of watchlist

            # Combine all candidates sorted by their score (RS for F&O, Gap for gaps)
            all_candidates = gap_stocks + fno_momentum
            all_candidates.sort(key=lambda x: x[1], reverse=True)

            for candidate in all_candidates:
                ticker = candidate[0]
                if ticker in {s[0] for s in selected}:
                    continue  # deduplicate

                sector = FNO_SECTORS.get(ticker, "Other")

                # Allow bypassing sector limit if it's a massive gap (alpha event)
                is_massive_gap = candidate[5].startswith("gap") and candidate[1] > 0.02

                if (
                    sector_counts.get(sector, 0) < max_per_sector
                    or sector == "Other"
                    or is_massive_gap
                ):
                    selected.append(candidate)
                    if sector != "Other":
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1

                if len(selected) >= self.top_n:
                    break

            # Final fallback: if still empty, use raw F&O list
            if not selected:
                logger.warning("No quality stocks found. Falling back to F&O top-N.")
                return FNO_UNIVERSE[: self.top_n]

            top_stocks = [s[0] for s in selected]

            logger.info(f"Screener identified {len(top_stocks)} 'Stocks in Play':")
            for ticker, sort_val, gap_pct, vol, price, source in selected:
                vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K"
                if source.startswith("gap") or source == "fno_gap":
                    logger.info(
                        f"  {ticker:<15} Gap: {gap_pct*100:+.2f}%  AvgVol: {vol_str}  Price: ₹{price:.1f}  [{source}]"
                    )
                else:
                    logger.info(
                        f"  {ticker:<15} Mom: {sort_val*100:+.2f}%  AvgVol: {vol_str}  Price: ₹{price:.1f}  [{source}]"
                    )

            return top_stocks

        except Exception as e:
            logger.error(f"Dynamic screener encountered an error: {e}")
            import traceback

            traceback.print_exc()
            logger.info("Falling back to F&O top-N.")
            return FNO_UNIVERSE[: self.top_n]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    screener = DynamicScreener(top_n=10)
    screener.scan_pre_market()
