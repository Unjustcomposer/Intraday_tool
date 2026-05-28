# Intraday Quant System — Antigravity Task File & Prompt

## Project Goal
Replicate the NSE/BSE Institutional-Grade Intraday Quant Trading System as a
fully functional, modular Python application. The system covers data ingestion,
feature engineering, regime detection, alpha modeling, meta-labeling, ensemble
scoring, risk management, position sizing, execution simulation, and backtesting.

---

## Antigravity Prompt

```
Build a production-ready, modular intraday quantitative trading system for
NSE/BSE Indian equities in Python. The system must replicate the following
architecture exactly. Use a clean package structure under the root folder
`intraday_quant_system/`. Each module must be independently importable and
testable. Use type hints throughout. Follow the spec below precisely.

SYSTEM OVERVIEW:
- Multi-factor, regime-adaptive, meta-labeled intraday trading system
- Primary exchange: NSE (National Stock Exchange, India)
- Data granularity: tick, 1-min, 5-min, daily
- Trading hours: 9:15 AM to 3:15 PM IST
- Hard exit: all positions closed by 3:15 PM
- Base currency: INR

---

MODULE 1 — data/
Build data ingestion and storage.

1a. market_data.py
  - Connect to Zerodha Kite API (zerodha/kiteconnect)
  - Fetch OHLCV + VWAP + bid/ask/depth + OI + spread + trade_count +
    aggressor_side for each symbol
  - Timeframes: tick, 1min, 5min, daily
  - Store raw data in TimescaleDB (hypertable: symbol, timestamp)
  - Cache hot data in Redis (TTL 60 seconds)
  - Output: pd.DataFrame with columns:
    [symbol, open, high, low, close, volume, vwap, bid_price, ask_price,
     bid_volume, ask_volume, oi, spread, trade_count, aggressor_side]

1b. fundamental_data.py
  - Universe filter function: apply_universe_filter(df) -> df
  - Filters:
      market_cap > 500 Cr INR
      avg_daily_turnover > 20 Cr INR
      debt_to_equity < 3
      piotroski_score >= 5
      altman_z_score > 1.8
  - Features to compute: ROE, ROCE, gross_margin, operating_margin,
    interest_coverage, revenue_growth, profit_growth, eps_growth, fcf_growth,
    PE, PB, EV/EBITDA, PEG, promoter_change, fii_change, dii_change,
    analyst_upgrades

1c. news_data.py
  - Fetch news from configured RSS/API feeds
  - Store raw article text with symbol tag and timestamp
  - Output: List[dict] with keys [headline, body, symbol, published_at]

---

MODULE 2 — features/
Build the feature engineering pipeline. Five orthogonal clusters.

2a. flow_features.py  (MOST IMPORTANT cluster)
  Functions:
  - relative_volume(df) -> Series        # RVOL = CurrentVol / AvgVol(20)
  - vwap_deviation(df) -> Series         # (price - vwap) / vwap
  - order_imbalance(df) -> Series        # (bid_vol - ask_vol) / (bid_vol + ask_vol)
  - volume_delta(df) -> Series           # cumulative buy_vol - sell_vol
  - microprice(df) -> Series             # (bid*ask_vol + ask*bid_vol)/(bid_vol+ask_vol)
  - trade_aggression(df) -> Series       # ratio of market orders to total

2b. trend_features.py
  Functions:
  - relative_strength(stock_ret, index_ret) -> Series   # stock_ret - index_ret
  - ema_slope(df, period=20) -> Series
  - adx(df, period=14) -> Series
  - momentum(df, period=10) -> Series
  - sector_strength(df, sector_ret) -> Series
  - beta(stock_ret, index_ret, window=60) -> Series

2c. volatility_features.py
  Functions:
  - atr(df, period=14) -> Series
  - realized_volatility(df, window=20) -> Series
  - bollinger_width(df, period=20) -> Series
  - volatility_percentile(df, lookback=252) -> Series   # 0-100
  - range_expansion(df) -> Series                        # (H-L)/ATR

2d. sentiment_features.py
  Functions:
  - encode_sentiment(finbert_output) -> Series   # float -1 to 1
  - earnings_surprise_score(actual, estimate) -> float
  - guidance_change_score(event_text) -> float   # via FinBERT
  - event_severity(event_type: str) -> float     # lookup table

2e. market_context_features.py
  Functions:
  - nifty_trend(nifty_df) -> str     # 'up' | 'down' | 'sideways'
  - vix_level(vix_df) -> float
  - market_breadth(universe_df) -> float   # advance/decline ratio
  - usd_inr(fx_df) -> float
  - crude_oil_price() -> float
  - sector_rotation_score(sector_returns) -> dict

2f. feature_store.py
  - FeatureStore class
  - Methods: compute_all(symbol, df) -> pd.DataFrame
  - Persist computed features to Parquet (partitioned by date/symbol)
  - Load features for backtesting: load(symbol, start, end) -> pd.DataFrame

---

MODULE 3 — regime/
Build the market regime detection engine.

3a. hmm_regime.py
  - RegimeDetector class using hmmlearn (GaussianHMM, n_components=5)
  - Input features: vix, breadth, atr_percentile, correlation_dispersion,
    index_slope, market_volume
  - fit(df) -> self
  - predict(df) -> pd.Series of regime labels:
      0: 'low_vol_trend'
      1: 'high_vol_trend'
      2: 'mean_reverting'
      3: 'crisis'
      4: 'sector_rotation'
  - Regime-to-exposure map:
      low_vol_trend:   1.00
      high_vol_trend:  0.60
      mean_reverting:  0.40
      crisis:          0.10
      sector_rotation: 0.70

3b. autoencoder_regime.py  (advanced version)
  - PyTorch autoencoder that compresses regime features to latent space
  - KMeans clustering on latent vectors (k=5)
  - encode(df) -> latent_df
  - cluster(latent_df) -> labels
  - Map cluster labels to named regimes using silhouette analysis

---

MODULE 4 — models/
Build the primary alpha model.

4a. lgbm_model.py
  - LGBMAlphaModel class
  - Label function: make_labels(df, atr_mult_up=2, atr_mult_down=1,
      horizon_minutes=45) -> pd.Series
    * Label = 1 if stock moves +2 ATR before -1 ATR within 45 minutes
    * Label = 0 otherwise
  - Parameters (must match exactly):
      objective='binary'
      metric='auc'
      num_leaves=127
      learning_rate=0.03
      n_estimators=1200
      feature_fraction=0.7
      bagging_fraction=0.8
      min_child_samples=100
      lambda_l1=0.1
      lambda_l2=1.0
  - Methods:
      train(X, y) -> self
      predict_proba(X) -> np.ndarray
      feature_importance() -> pd.DataFrame  # sorted SHAP or gain
      save(path) / load(path)

4b. catboost_meta_labeler.py
  - MetaLabeler class (CatBoostClassifier)
  - Input: primary model predictions + all features
  - Label: 1 if the primary model's trade was actually profitable
  - Methods:
      train(X_primary_preds, X_features, y_trade_outcome)
      predict_proba(X) -> np.ndarray   # confidence that trade is worth taking
      save(path) / load(path)

---

MODULE 5 — transformers/
Build the deep learning sequence model.

5a. temporal_transformer.py
  - PyTorch TemporalTransformer model
  - Input shape: (batch, 120, n_features)
  - Features per bar: OHLCV, RVOL, VWAP, ATR, OrderFlow,
    SectorReturn, IndexReturn  (= 11 features)
  - Architecture:
      EmbeddingLayer(11 -> 64)
      TransformerEncoder(nhead=4, num_layers=3, dim_feedforward=256)
      AttentionPooling
      Dense(64 -> 32 -> 1)
      Sigmoid output
  - Methods:
      forward(x) -> probability tensor
      train_epoch(dataloader, optimizer) -> float  # returns avg loss
      evaluate(dataloader) -> dict  # AUC, accuracy, etc.
      save(path) / load(path)
  - DataLoader: SequenceDataset class (sliding window, stride=1)

---

MODULE 6 — nlp/
Build the FinBERT NLP sentiment engine.

6a. finbert_sentiment.py
  - Load pretrained FinBERT model (ProsusAI/finbert from HuggingFace)
  - SentimentEngine class:
      analyze(text: str) -> dict
        Returns: {sentiment: str, score: float, label: int}
      analyze_batch(texts: List[str]) -> pd.DataFrame
  - EventClassifier class:
      classify(headline: str) -> str
        Returns one of: 'earnings', 'guidance', 'merger', 'regulatory',
          'macro', 'rating_change', 'insider', 'other'
      severity(event_type: str, sentiment_score: float) -> float

6b. news_velocity.py
  - compute_velocity(articles: List[dict], window_minutes=30) -> float
    * Articles per minute for the symbol in the rolling window
  - social_sentiment_score(symbol, lookback_minutes=60) -> float

---

MODULE 7 — meta_labeling/
Build the meta-labeling pipeline.

7a. meta_pipeline.py
  - MetaLabelingPipeline class:
      run(primary_model, catboost_meta, X_features) -> pd.DataFrame
        Columns: [signal, primary_prob, meta_prob, take_trade]
      generate_trade_outcomes(df, price_df) -> pd.Series
        * Simulate outcome of each primary signal for training meta model
      retrain(X, y_outcomes) -> self

---

MODULE 8 — signals/
Build the ensemble and filtering engine.

8a. ensemble.py
  - EnsembleScorer class
  - Weights (must match exactly):
      lgbm_weight       = 0.40
      transformer_weight = 0.20
      meta_weight       = 0.20
      sentiment_weight  = 0.15
      regime_weight     = 0.05
  - compute_score(lgbm_prob, transformer_prob, meta_prob,
      sentiment_score, regime_score) -> float
  - Signal thresholds:
      score > 0.82  -> 'strong_buy'
      score > 0.72  -> 'buy'
      score > 0.62  -> 'small_position'
      else          -> 'no_trade'

8b. filters.py
  - LiquidityFilter: adv > 20Cr AND spread < 0.15%
  - RvolFilter: rvol > 1.8
  - RegimeFilter: block trades in low_vol_chop regime
  - StructureFilter: block at major resistance, thin liquidity, event uncertainty
  - EntryFilter.check_all(features_dict) -> bool
    * Checks: rvol>1.8, price>vwap, sector_strength>0,
      sentiment>0.6, spread<0.15%, ensemble_score>threshold

---

MODULE 9 — risk/
Build the risk management layer.

9a. position_sizing.py
  - kelly_fraction(win_prob, win_loss_ratio) -> float
  - volatility_adjusted_size(target_vol, asset_vol, capital) -> float
    * PositionSize = (target_vol / asset_vol) * capital
  - PortfolioLimits (enforce all at once):
      max_risk_per_trade   = 0.02    # 2% of capital
      max_open_positions   = 5
      max_sector_exposure  = 0.25    # 25%
      max_portfolio_exposure = 0.70  # 70%
      min_cash_reserve     = 0.30    # 30%

9b. stop_loss.py
  - ATRStopEngine class:
      compute_stop(entry_price, atr, regime) -> float
        SL = entry - k * ATR
        k values by regime:
          low_vol_trend:   1.2
          high_vol_trend:  1.5   (trending)
          mean_reverting:  2.0   (high vol)
          crisis:          2.5
      compute_target(entry_price, risk, confidence) -> float
        Target = entry + RR * risk
        RR by confidence:
          high:   2.5
          medium: 2.0
          low:    1.5

9c. portfolio_risk.py
  - PortfolioRiskMonitor class:
      check(portfolio_state) -> dict  # {action: str, reason: str}
      Kill switch rules:
        daily_loss  > 3%  -> stop_trading=True
        weekly_loss > 6%  -> reduce_size_50_pct=True
        drawdown    > 10% -> pause_review=True
        vix         > 25  -> cut_exposure=True
  - VaR computation:
      var_historical(returns, confidence=0.95) -> float
      var_parametric(mu, sigma, confidence=0.95) -> float

---

MODULE 10 — execution/
Build the smart execution engine.

10a. execution_engine.py
  - ExecutionEngine class (connects to Zerodha Kite API)
  - Order routing logic:
      if confidence == 'strong_buy':   place MARKET order
      elif spread < 0.15%:             place passive LIMIT order
      elif spread > 0.30%:             place ICEBERG / staggered order
      else:                            avoid / no order
  - Slippage model:
      estimate_slippage(order_size, adv) -> float
      slippage = k * sqrt(order_size / adv)   # k=0.1 default
  - Methods:
      place_order(symbol, side, qty, order_type, price=None)
      cancel_order(order_id)
      get_positions() -> pd.DataFrame
      get_order_history() -> pd.DataFrame

10b. order_manager.py
  - Track all open orders and positions
  - Hard exit trigger: close all positions by 3:15 PM IST
  - Stop-loss monitor: run every tick, trigger stop if price hits SL
  - Daily trade counter: enforce max 5 trades per day

---

MODULE 11 — backtesting/
Build the backtesting framework.

11a. backtest_engine.py
  - BacktestEngine class (use VectorBT internally)
  - Mandatory anti-bias requirements:
      no_lookahead=True       # strict — no future data leakage
      no_survivorship=True    # use full historical universe
      realistic_slippage=True # use slippage model from execution/
      realistic_costs=True    # brokerage + STT + stamp duty + SEBI fees
  - run(features_df, signals_df, price_df) -> BacktestResult
  - BacktestResult attributes:
      equity_curve, drawdown_series, trade_log,
      sharpe, sortino, profit_factor, max_drawdown,
      win_rate, expectancy, regime_breakdown

11b. walk_forward.py
  - WalkForwardValidator class:
      training_window = 252  # trading days
      validation_window = 63
      step_size = 21
  - run(full_df, model_factory) -> List[BacktestResult]
  - aggregate_results(results) -> dict  # combined metrics

11c. monte_carlo.py
  - MonteCarloStressTester class:
      n_simulations = 1000
      randomize: slippage (±50%), latency (0–50ms),
                 fill_quality (80–100%), volatility (×0.5 to ×2)
  - run(backtest_result) -> StressTestResult
  - StressTestResult: p5_sharpe, p50_sharpe, p95_sharpe,
      p5_drawdown, p50_drawdown, worst_case_ruin_probability

---

MODULE 12 — monitoring/
Build the live monitoring layer.

12a. dashboard.py
  - Streamlit app: streamlit run monitoring/dashboard.py
  - Pages:
      Live P&L + equity curve
      Open positions + risk exposure
      Signal log with ensemble scores
      Feature drift alerts
      Regime state indicator
      Daily kill switch status

12b. drift_detector.py
  - FeatureDriftMonitor class:
      fit_baseline(X_train) -> self   # store training distributions
      detect(X_live) -> dict          # {feature: drift_score}
      alert_threshold = 2.5           # z-score for alert
  - SignalDecayMonitor class:
      monitor(predictions, outcomes, window=63) -> float  # rolling AUC
      decay_alert_threshold = 0.52    # below this = model degraded

12c. grafana_exporter.py
  - Expose Prometheus metrics endpoint (/metrics) for Grafana scraping
  - Metrics to export:
      quant_daily_pnl, quant_open_positions, quant_ensemble_score,
      quant_drawdown, quant_regime, quant_vix, quant_signal_auc

---

MODULE 13 — deployment/
Orchestration and configuration.

13a. config.py
  - QuantConfig dataclass (load from config.yaml):
      zerodha_api_key, zerodha_api_secret
      timescaledb_url, redis_url
      universe: List[str]       # NSE symbols
      target_volatility: float  # 0.15 default
      max_capital: float        # INR
      trading_start: "09:15"
      trading_end:   "15:15"
      retraining_frequency: "weekly"

13b. pipeline_runner.py
  - Main orchestration loop:
      1. On market open: run regime detection
      2. Every 1 minute: compute features for universe
      3. Every 5 minutes: run alpha models + ensemble
      4. On signal: check filters -> size position -> execute
      5. Every tick: monitor stops
      6. At 15:10: begin closing all positions
      7. At 15:15: hard close + log + retrain check
  - Use APScheduler for cron jobs
  - Graceful shutdown on SIGINT

---

TECHNOLOGY STACK (must use exactly):
  Market Data:    zerodha/kiteconnect
  Storage:        TimescaleDB (psycopg2/asyncpg) + Apache Parquet (pyarrow)
  Cache:          Redis (redis-py)
  ML:             lightgbm, catboost
  Deep Learning:  PyTorch (torch, transformers)
  NLP:            transformers (HuggingFace, ProsusAI/finbert)
  Regime:         hmmlearn, scikit-learn
  Backtesting:    vectorbt
  Dashboard:      streamlit
  Monitoring:     grafana + prometheus-client
  API:            fastapi + uvicorn
  Scheduler:      APScheduler
  Config:         pydantic, python-dotenv, PyYAML
  Testing:        pytest, pytest-asyncio

---

PROJECT STRUCTURE (create exactly):
  intraday_quant_system/
  ├── config.yaml
  ├── requirements.txt
  ├── README.md
  ├── data/
  │   ├── market_data.py
  │   ├── fundamental_data.py
  │   └── news_data.py
  ├── features/
  │   ├── flow_features.py
  │   ├── trend_features.py
  │   ├── volatility_features.py
  │   ├── sentiment_features.py
  │   ├── market_context_features.py
  │   └── feature_store.py
  ├── regime/
  │   ├── hmm_regime.py
  │   └── autoencoder_regime.py
  ├── models/
  │   ├── lgbm_model.py
  │   └── catboost_meta_labeler.py
  ├── transformers/
  │   └── temporal_transformer.py
  ├── nlp/
  │   ├── finbert_sentiment.py
  │   └── news_velocity.py
  ├── meta_labeling/
  │   └── meta_pipeline.py
  ├── signals/
  │   ├── ensemble.py
  │   └── filters.py
  ├── risk/
  │   ├── position_sizing.py
  │   ├── stop_loss.py
  │   └── portfolio_risk.py
  ├── execution/
  │   ├── execution_engine.py
  │   └── order_manager.py
  ├── backtesting/
  │   ├── backtest_engine.py
  │   ├── walk_forward.py
  │   └── monte_carlo.py
  ├── monitoring/
  │   ├── dashboard.py
  │   ├── drift_detector.py
  │   └── grafana_exporter.py
  ├── snn/                        # experimental neuromorphic module
  │   └── snn_order_flow.py       # spiking neural net for order flow anomalies
  └── deployment/
      ├── config.py
      └── pipeline_runner.py

---

TESTING REQUIREMENTS:
  - Unit tests for every public function in features/, risk/, signals/
  - Integration test: run full pipeline on 5 days of synthetic OHLCV data
  - Backtest smoke test: run WalkForwardValidator on 1 year of data
  - All tests in tests/ directory mirroring the module structure
  - pytest --cov target: 80% minimum coverage

---

CODING STANDARDS:
  - Python 3.11+
  - Type hints on all function signatures
  - Docstrings on all public classes and methods (Google style)
  - No hardcoded credentials — all secrets via environment variables
  - Logging via Python logging module (structured JSON logs in production)
  - All DataFrames must have datetime index in IST timezone (Asia/Kolkata)
  - No lookahead bias anywhere — enforce with strict timestamp checks

Build each module completely, with working imports and no placeholder stubs.
Start with: requirements.txt, config.yaml, then modules in this order:
features/ → regime/ → models/ → signals/ → risk/ → execution/ → backtesting/ → monitoring/ → deployment/
```

---

## Task Breakdown (copy into Antigravity task list)

### Phase 1 — Foundation
- [ ] 1.1 Create project structure (all directories + `__init__.py` files)
- [ ] 1.2 Write `requirements.txt` with pinned versions
- [ ] 1.3 Write `config.yaml` with all parameters
- [ ] 1.4 Write `deployment/config.py` (QuantConfig dataclass)
- [ ] 1.5 Set up `.env.example` with all required environment variables

### Phase 2 — Data Layer
- [ ] 2.1 `data/market_data.py` — Kite API + TimescaleDB + Redis cache
- [ ] 2.2 `data/fundamental_data.py` — universe filter + fundamental features
- [ ] 2.3 `data/news_data.py` — RSS/API news fetch + symbol tagging

### Phase 3 — Feature Engineering
- [ ] 3.1 `features/flow_features.py` — RVOL, VWAP dev, OI, delta, microprice
- [ ] 3.2 `features/trend_features.py` — RS, EMA slope, ADX, momentum, beta
- [ ] 3.3 `features/volatility_features.py` — ATR, realized vol, Bollinger width
- [ ] 3.4 `features/sentiment_features.py` — FinBERT output encoder, event severity
- [ ] 3.5 `features/market_context_features.py` — Nifty trend, VIX, breadth
- [ ] 3.6 `features/feature_store.py` — compute_all + Parquet persistence

### Phase 4 — Regime Detection
- [ ] 4.1 `regime/hmm_regime.py` — GaussianHMM (5 regimes) + exposure map
- [ ] 4.2 `regime/autoencoder_regime.py` — PyTorch AE + KMeans clustering

### Phase 5 — Alpha Models
- [ ] 5.1 `models/lgbm_model.py` — label function + LightGBM with exact params
- [ ] 5.2 `models/catboost_meta_labeler.py` — meta-label training + inference
- [ ] 5.3 `transformers/temporal_transformer.py` — PyTorch Transformer (120-bar)
- [ ] 5.4 `nlp/finbert_sentiment.py` — FinBERT loader + SentimentEngine
- [ ] 5.5 `nlp/news_velocity.py` — velocity + social sentiment score
- [ ] 5.6 `meta_labeling/meta_pipeline.py` — full meta-label orchestration

### Phase 6 — Signal Engine
- [ ] 6.1 `signals/ensemble.py` — EnsembleScorer with exact weights + thresholds
- [ ] 6.2 `signals/filters.py` — Liquidity, RVOL, Regime, Structure, EntryFilter

### Phase 7 — Risk Management
- [ ] 7.1 `risk/position_sizing.py` — Kelly + volatility targeting + limits
- [ ] 7.2 `risk/stop_loss.py` — ATRStopEngine (k by regime) + RR targets
- [ ] 7.3 `risk/portfolio_risk.py` — kill switches + VaR computation

### Phase 8 — Execution
- [ ] 8.1 `execution/execution_engine.py` — Kite order routing + slippage model
- [ ] 8.2 `execution/order_manager.py` — position tracking + 3:15 PM hard exit

### Phase 9 — Backtesting
- [ ] 9.1 `backtesting/backtest_engine.py` — VectorBT wrapper, no-lookahead checks
- [ ] 9.2 `backtesting/walk_forward.py` — 252/63/21 day walk-forward
- [ ] 9.3 `backtesting/monte_carlo.py` — 1000-simulation stress tester

### Phase 10 — Monitoring
- [ ] 10.1 `monitoring/dashboard.py` — Streamlit live dashboard (6 pages)
- [ ] 10.2 `monitoring/drift_detector.py` — feature drift + signal decay monitor
- [ ] 10.3 `monitoring/grafana_exporter.py` — Prometheus metrics endpoint

### Phase 11 — Orchestration
- [ ] 11.1 `deployment/pipeline_runner.py` — full APScheduler orchestration loop
- [ ] 11.2 `snn/snn_order_flow.py` — experimental SNN for order flow anomalies

### Phase 12 — Tests
- [ ] 12.1 Unit tests for all feature functions (synthetic data)
- [ ] 12.2 Unit tests for risk/signals/ensemble modules
- [ ] 12.3 Integration test: 5-day synthetic pipeline end-to-end
- [ ] 12.4 Backtest smoke test: 1-year walk-forward run
- [ ] 12.5 Achieve ≥80% pytest-cov coverage

---

## Key Implementation Notes for Antigravity

| Decision Point | Spec |
|---|---|
| Label design | +2 ATR before −1 ATR within 45 min — NOT binary up/down |
| Primary model | LightGBM (fastest inference, explainable, handles noisy tabular) |
| Sequence model | Transformer, NOT LSTM (better long-range attention over 120 bars) |
| Regime model | GaussianHMM (primary), Autoencoder+KMeans (advanced) |
| Meta model | CatBoost (handles categorical regime features well) |
| Execution default | Passive limit orders — avoid market orders except strong_buy |
| Slippage formula | k × √(order_size / ADV), k=0.1 |
| Hard constraints | No lookahead, no survivorship bias, realistic costs always |
| Timezone | All timestamps in Asia/Kolkata (IST, UTC+5:30) |
| Exit discipline | All positions CLOSED by 3:15 PM IST regardless of P&L |
