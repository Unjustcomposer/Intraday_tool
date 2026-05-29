import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MetaLabelingPipeline:
    def __init__(self):
        self.take_trade_threshold = 0.5

    def run(self, primary_model, catboost_meta, X_features: pd.DataFrame) -> pd.DataFrame:
        """
        Columns: [signal, primary_prob, meta_prob, take_trade]
        """
        logger.info(f"Running MetaLabelingPipeline on {len(X_features)} samples")
        
        # 1. Primary Model Predictions
        primary_prob = primary_model.predict_proba(X_features)
        signal = (primary_prob > 0.5).astype(int)
        
        # 2. Meta Model Predictions
        meta_prob = catboost_meta.predict_proba(X_features, primary_preds=primary_prob)
        
        # 3. Decision
        take_trade = (meta_prob > self.take_trade_threshold).astype(int)
        
        results = pd.DataFrame({
            'signal': signal,
            'primary_prob': primary_prob,
            'meta_prob': meta_prob,
            'take_trade': take_trade
        }, index=X_features.index)
        
        return results

    def generate_trade_outcomes(self, df: pd.DataFrame, price_df: pd.DataFrame,
                                atr_mult_up: float = 2.0, atr_mult_down: float = 1.0,
                                horizon_bars: int = 45) -> pd.Series:
        """
        Generate trade outcome labels using triple-barrier labeling,
        matching the primary model's labeling logic (LGBMAlphaModel.make_labels).
        
        Outcome = 1 if primary model's predicted direction matches the actual
        triple-barrier label outcome. Outcome = 0 otherwise.
        
        Uses ATR-based barriers consistent with the primary model's configuration.
        """
        if not all(col in price_df.columns for col in ['close', 'atr']):
            logger.warning("price_df missing 'close' or 'atr' columns, returning all zeros")
            return pd.Series(0, index=df.index)
        
        if 'signal' not in df.columns:
            logger.warning("df missing 'signal' column, returning all zeros")
            return pd.Series(0, index=df.index)
        
        # Compute triple-barrier labels using the same logic as the primary model
        closes = price_df['close'].values
        atrs = price_df['atr'].values
        barrier_labels = np.full(len(price_df), np.nan)
        
        for i in range(len(price_df) - horizon_bars):
            current_price = closes[i]
            current_atr = atrs[i]
            
            if np.isnan(current_atr) or current_atr == 0:
                continue
            
            upper_barrier = current_price + (atr_mult_up * current_atr)
            lower_barrier = current_price - (atr_mult_down * current_atr)
            
            window = closes[i + 1 : i + 1 + horizon_bars]
            
            hit_upper = False
            hit_lower = False
            for price in window:
                if price >= upper_barrier:
                    hit_upper = True
                    break
                elif price <= lower_barrier:
                    hit_lower = True
                    break
            
            if hit_upper:
                barrier_labels[i] = 1   # Upward move confirmed
            elif hit_lower:
                barrier_labels[i] = -1  # Downward move confirmed
            else:
                barrier_labels[i] = 0   # No barrier hit (neutral)
        
        barrier_series = pd.Series(barrier_labels, index=price_df.index)
        
        # Align barrier labels with df index
        aligned_barriers = barrier_series.reindex(df.index)
        
        # Determine outcome: 1 if predicted direction matches actual barrier outcome
        outcomes = []
        for idx in df.index:
            barrier_label = aligned_barriers.get(idx, np.nan)
            if pd.isna(barrier_label):
                outcomes.append(0)
                continue
            
            signal = df.loc[idx, 'signal']
            
            # signal > 0 means the primary model predicted LONG
            # signal == 0 means the primary model did NOT predict long (neutral/no trade)
            # Outcome = 1 only if the model's predicted direction was correct
            if signal > 0 and barrier_label == 1:
                # Primary predicted long, and price hit upper barrier
                outcomes.append(1)
            elif signal < 0 and barrier_label == -1:
                # Primary predicted short, and price hit lower barrier
                outcomes.append(1)
            else:
                # Direction mismatch, neutral signal, or no barrier hit
                outcomes.append(0)
        
        return pd.Series(outcomes, index=df.index)

    def retrain(self, primary_model, catboost_meta, X: pd.DataFrame, y_outcomes: pd.Series):
        """Retrain meta model"""
        primary_prob = primary_model.predict_proba(X)
        catboost_meta.train(primary_prob, X, y_outcomes)
        return self
