import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_universe_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the universe of stocks based on fundamental criteria.
    Filters:
      market_cap > 500 Cr INR
      avg_daily_turnover > 20 Cr INR
      debt_to_equity < 3
      piotroski_score >= 5
      altman_z_score > 1.8
      
    Args:
        df: DataFrame containing fundamental data for stocks
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
        
    logger.info(f"Applying fundamental universe filter to {len(df)} stocks")
    
    # Fill missing values to avoid drop
    df = df.fillna(0)
    
    # Ensure required columns exist, default to values that would exclude if missing
    required_cols = {
        'market_cap': 0,
        'avg_daily_turnover': 0,
        'debt_to_equity': 999,
        'piotroski_score': 0,
        'altman_z_score': 0
    }
    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default
    
    # Apply filters
    filtered_df = df[
        (df['market_cap'] > 500) & 
        (df['avg_daily_turnover'] > 20) &
        (df['debt_to_equity'] < 3) &
        (df['piotroski_score'] >= 5) &
        (df['altman_z_score'] > 1.8)
    ]
    
    logger.info(f"Filtered down to {len(filtered_df)} stocks")
    return filtered_df

def compute_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes fundamental features for the universe.
    Features to compute: ROE, ROCE, gross_margin, operating_margin,
    interest_coverage, revenue_growth, profit_growth, eps_growth, fcf_growth,
    PE, PB, EV/EBITDA, PEG, promoter_change, fii_change, dii_change,
    analyst_upgrades
    """
    if df.empty:
        return df
        
    logger.info("Computing fundamental features")
    features = df.copy()
    
    # Helper to safely get a column or return a Series of default values
    def _col(name: str, default: float = 0.0) -> pd.Series:
        if name in features.columns:
            return features[name]
        return pd.Series(default, index=features.index)
    
    def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator / denominator.replace(0, np.nan)
    
    # Profitability Ratios
    features['roe'] = _safe_div(_col('net_income'), _col('shareholder_equity', 1))
    features['roce'] = _safe_div(_col('ebit'), _col('capital_employed', 1))
    features['gross_margin'] = _safe_div(_col('gross_profit'), _col('revenue', 1))
    features['operating_margin'] = _safe_div(_col('operating_income'), _col('revenue', 1))
    
    # Solvency & Growth
    features['interest_coverage'] = _safe_div(_col('ebit'), _col('interest_expense', 1))
    
    # Valuation Metrics
    features['pe'] = _safe_div(_col('price'), _col('eps', 1))
    features['pb'] = _safe_div(_col('price'), _col('book_value_per_share', 1))
    features['ev_ebitda'] = _safe_div(_col('enterprise_value'), _col('ebitda', 1))
    
    # Growth metrics are typically pre-computed YoY
    
    return features
