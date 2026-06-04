import logging
import yfinance as yf
import pandas as pd
from typing import List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Full NSE F&O Universe watchlist (deduplicated)
FNO_UNIVERSE = [
    "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "TATAMOTORS.NS", "WIPRO.NS", "BAJFINANCE.NS", "LT.NS",
    "HCLTECH.NS", "BHARTIARTL.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "ADANIENT.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "TECHM.NS", "INDUSINDBK.NS", "HINDALCO.NS",
    "DRREDDY.NS", "CIPLA.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS",
    "ASIANPAINT.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BPCL.NS", "BRITANNIA.NS",
    "DIVISLAB.NS", "EICHERMOT.NS", "GRASIM.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDUNILVR.NS", "ITC.NS", "M&M.NS", "NESTLEIND.NS",
    "SBILIFE.NS", "TATACHEM.NS", "TATACONSUM.NS", "UPL.NS",
    "AMBUJACEM.NS", "ASHOKLEY.NS", "AUROPHARMA.NS", "BANDHANBNK.NS",
    "BANKBARODA.NS", "BHEL.NS", "BIOCON.NS", "BOSCHLTD.NS", "CANBK.NS",
    "CHOLAFIN.NS", "COFORGE.NS", "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS",
    "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DELTACORP.NS",
    "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS",
    "GLENMARK.NS", "GMRINFRA.NS", "GODREJCP.NS", "GODREJPROP.NS", "GRANULES.NS",
    "GUJGASLTD.NS", "HAL.NS", "HAVELLS.NS", "HDFCAMC.NS", "ICICIGI.NS",
    "ICICIPRULI.NS", "IDEA.NS", "IDFCFIRSTB.NS", "IEX.NS", "IGL.NS",
    "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIGO.NS", "INDUSTOWER.NS",
    "IPCALAB.NS", "IRCTC.NS", "JINDALSTEL.NS", "JUBLFOOD.NS", "L&TFH.NS",
    "LALPATHLAB.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LTIM.NS", "LTTS.NS",
    "M&MFIN.NS", "MANAPPURAM.NS", "MARICO.NS", "MGL.NS", "MOTHERSON.NS",
    "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS",
    "NAVINFLUOR.NS", "OBEROIRLTY.NS", "OFSS.NS", "PAGEIND.NS", "PEL.NS",
    "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS",
    "PNB.NS", "POLYCAB.NS", "PVRINOX.NS", "RAMCOCEM.NS", "RBLBANK.NS",
    "RECLTD.NS", "SAIL.NS", "SBICARD.NS", "SHREECEM.NS", "SIEMENS.NS",
    "SRF.NS", "SUNTV.NS", "SYNGENE.NS", "TATACOMM.NS",
    "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS",
    "VEDL.NS", "VOLTAS.NS", "ZEEL.NS", "ZYDUSLIFE.NS"
]

class DynamicScreener:
    """
    Scans the F&O universe at market open to identify "Stocks in Play".
    Uses threaded yfinance downloads to minimize latency.
    """
    def __init__(self, top_n: int = 20):
        self.top_n = top_n
        self.universe = FNO_UNIVERSE

    def scan_pre_market(self) -> List[str]:
        """
        Scan for top gap-ups and gap-downs.
        Returns a list of `top_n` symbols.
        """
        logger.info(f"Running dynamic pre-market screener on {len(self.universe)} F&O stocks...")
        
        # Download 2 days of daily data for the whole universe to calculate gaps
        # Use threads=True for parallel fetching, which cuts down latency significantly.
        try:
            # Setting progress=False prevents messy console output
            df = yf.download(
                self.universe, 
                period='5d', 
                interval='1d', 
                threads=True, 
                progress=False
            )
            
            if df.empty or 'Close' not in df.columns or 'Open' not in df.columns:
                logger.error("Screener failed to download data. Falling back to default top 20.")
                return self.universe[:self.top_n]
                
            # yfinance returns MultiIndex columns when downloading multiple tickers: (PriceType, Ticker)
            closes = df['Close']
            opens = df['Open']
            
            # Drop columns with all NaNs (e.g. delisted or not found)
            closes = closes.dropna(axis=1, how='all')
            opens = opens.dropna(axis=1, how='all')
            
            gaps = []
            
            for ticker in closes.columns:
                try:
                    # Drop individual NaNs to get the last two valid trading days
                    ticker_closes = closes[ticker].dropna()
                    ticker_opens = opens[ticker].dropna()
                    
                    if len(ticker_closes) >= 2 and len(ticker_opens) >= 1:
                        # Prev close is the second to last daily close
                        prev_close = ticker_closes.iloc[-2]
                        # Current open is the last daily open
                        curr_open = ticker_opens.iloc[-1]
                        
                        if prev_close > 0:
                            gap_pct = (curr_open - prev_close) / prev_close
                            gaps.append((ticker, gap_pct))
                except Exception as e:
                    logger.debug(f"Error calculating gap for {ticker}: {e}")
                    continue
                    
            if not gaps:
                logger.warning("No gaps calculated. Falling back to default top 20.")
                return self.universe[:self.top_n]
                
            # Sort by absolute gap percentage (highest absolute gap = most "in play")
            # We want both extreme gap-ups and extreme gap-downs.
            gaps.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Select the top N
            top_stocks = [ticker for ticker, gap in gaps[:self.top_n]]
            
            logger.info(f"Screener identified {len(top_stocks)} 'Stocks in Play':")
            for ticker, gap in gaps[:self.top_n]:
                logger.info(f"  {ticker:<15} Gap: {gap*100:+.2f}%")
                
            return top_stocks
            
        except Exception as e:
            logger.error(f"Dynamic screener encountered an error: {e}")
            logger.info("Falling back to default top 20.")
            return self.universe[:self.top_n]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    screener = DynamicScreener(top_n=10)
    screener.scan_pre_market()
