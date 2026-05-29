<![CDATA[<div align="center">

# ⚡ Institutional-Grade Intraday Quant Trading System

### NSE / BSE · Indian Equities · Fully Automated

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-Core_Engine-B7410E?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML_Models-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Redis](https://img.shields.io/badge/Redis-Pub%2FSub-DC382D?logo=redis&logoColor=white)](https://redis.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready, multi-process quantitative trading pipeline for Indian intraday equities. Combines calibrated ML ensembles, NLP-driven sentiment analysis, realistic execution simulation, and institutional-grade risk controls — architected for live capital deployment on NSE/BSE via Zerodha Kite.

</div>

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Backtesting](#-backtesting)
- [Testing](#-testing)
- [Monitoring](#-monitoring)
- [Disclaimer](#-disclaimer)

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROCESS A — Data Ingestion                  │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────────┐ │
│  │  Kite WebSocket │→│ Bar Aggregator │→│ Redis Pub/Sub (1-min)  │ │
│  └──────────────┘   └──────────────┘   └─────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  Completed bars published
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PROCESS B — Inference Pipeline                  │
│                                                                     │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐           │
│  │ Feature  │→│  Regime   │→│ ML Models │→│ Ensemble  │            │
│  │  Store   │  │ Detector │  │LGB/CB/TFT│  │  Scorer   │           │
│  └─────────┘  └──────────┘  └──────────┘  └─────┬─────┘           │
│                                                   │                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────▼──────────────┐  │
│  │ FinBERT  │→│ Sentiment  │→│    Signal Generation & Gating   │  │
│  │   NLP    │  │  Features  │  │  (Conformal Prediction Thresh) │  │
│  └──────────┘  └───────────┘  └──────────────────┬──────────────┘  │
│                                                   │                 │
│  ┌───────────────┐  ┌────────────┐  ┌────────────▼────────────┐   │
│  │ Position Sizer │→│ Risk Mgmt  │→│   Execution Engine      │   │
│  │ (Kelly/Vol)    │  │ (DPO/VaR)  │  │ (TWAP + Almgren-Chriss)│   │
│  └───────────────┘  └────────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🤖 Machine Learning & Signal Generation
- **Multi-model ensemble** — LightGBM, CatBoost (meta-labeler), TabNet, Temporal Fusion Transformer (TFT)
- **Platt-calibrated probabilities** — Sigmoid calibration on raw model outputs to prevent overconfidence
- **Inductive Conformal Prediction (ICP)** — Statistically rigorous, dynamic probability thresholds replacing hardcoded rules
- **Regime-conditional weights** — Ensemble weights shift dynamically across quiet, volatile, and crisis regimes
- **Walk-forward cross-validation** — Prevents data leakage; uses last fold model only (no refit on full data)

### 📰 NLP & Sentiment
- **FinBERT sentiment analysis** — Transformer-based financial sentiment scoring on live news headlines
- **Pre-market decoupled processing** — NLP runs before market open to avoid latency during trading hours
- **News kill switch** — Automatically disables trading on assets affected by severe market shocks
- **Knowledge graph** — Entity-relationship mapping for sector/stock event propagation

### 📊 Feature Engineering (60+ Features)
- **Flow features** — Order flow imbalance (OFI), trade flow, cumulative delta
- **L2 microstructure** — Bid-ask spread, order book imbalance (OBI), depth ratio, weighted mid-price
- **Volatility features** — Parkinson, Garman-Klass, Yang-Zhang estimators, ATR regime
- **Trend features** — Multi-timeframe EMA crossovers, ADX, momentum oscillators
- **Market context** — VIX proxy, sector-relative strength, put-call ratio
- **Options Greeks** — Black-Scholes Δ, Γ, Θ, Vega for hedging signals

### ⚙️ Execution & Market Impact
- **TWAP order chunking** — Splits institutional-sized orders to minimize market footprint
- **Almgren-Chriss slippage model** — Asymmetric market impact based on trade size, volatility, and ADV
- **VWAP fill simulation** — Realistic backtesting fills using volume-weighted average price
- **Queue position penalties** — Simulates realistic queue depth and fill delays
- **Exponential backoff** — Resilient API retry logic with jitter for connection failures

### 🛡 Risk Management
- **Dynamic position sizing** — Volatility-scaled Kelly criterion with regime adjustments
- **Differentiable Portfolio Optimization (DPO)** — Convex optimization layer for portfolio construction
- **Multi-level circuit breakers** — Daily loss limit (3%), weekly limit (6%), max drawdown (10%)
- **Sector exposure limits** — 25% max per sector, 70% max portfolio exposure
- **Hard exit enforcement** — 3:15 PM IST forced liquidation of all open positions
- **Margin reconciliation** — Real-time margin tracking and leveraged capital validation

### 🏎 Infrastructure
- **Rust core engine** — Performance-critical tick processing compiled via PyO3/Maturin
- **Multi-process architecture** — `ProcessPoolExecutor` for non-blocking ML inference
- **Redis pub/sub** — Real-time tick-to-bar message passing between processes
- **TimescaleDB** — PostgreSQL extension optimized for time-series financial data
- **Docker Compose** — Full stack containerization (pipeline, Redis, TimescaleDB, Prometheus, Grafana)
- **Prometheus + Grafana** — Real-time system metrics, latency tracking, and alerting dashboards

---

## 📁 Project Structure

```
intraday_quant_system/
│
├── backtesting/                  # Backtesting framework
│   ├── backtest_engine.py        # VWAP fills, Almgren-Chriss slippage, queue penalties
│   ├── cpcv_validator.py         # Combinatorial Purged Cross-Validation
│   ├── walk_forward.py           # Walk-forward out-of-sample validation
│   └── monte_carlo.py            # Monte Carlo simulation for P&L distributions
│
├── core_engine/                  # Rust-powered tick processing (PyO3)
│   ├── Cargo.toml
│   └── src/
│
├── data/                         # Data layer
│   ├── market_data.py            # Kite API market data fetcher
│   ├── bar_aggregator.py         # Tick-to-bar aggregation + Redis publish
│   ├── websocket_ticker.py       # Live WebSocket tick consumer
│   ├── news_data.py              # RSS/API news ingestion
│   ├── fundamental_data.py       # Fundamental data fetcher
│   └── nse_calendar.py           # NSE trading calendar & holidays
│
├── deployment/                   # Pipeline orchestration
│   ├── pipeline_runner.py        # Main inference loop (Process B)
│   └── config.py                 # YAML config loader with validation
│
├── execution/                    # Order execution
│   ├── execution_engine.py       # TWAP chunking, circuit limits, velocity checks
│   ├── order_manager.py          # Order lifecycle management
│   ├── algorithms.py             # Execution algorithm implementations
│   ├── reconciliation.py         # Margin & position reconciliation
│   └── post_trade_analyzer.py    # Post-trade execution quality analysis
│
├── features/                     # Feature engineering (60+ features)
│   ├── feature_store.py          # Central feature computation pipeline
│   ├── flow_features.py          # Order flow imbalance, cumulative delta
│   ├── l2_microstructure.py      # L2 book metrics (OBI, spread, depth)
│   ├── volatility_features.py    # Parkinson, GK, YZ volatility estimators
│   ├── trend_features.py         # EMA crossovers, ADX, momentum
│   ├── time_features.py          # Intraday time-of-day features
│   ├── market_context_features.py# VIX proxy, sector strength, PCR
│   ├── sentiment_features.py     # NLP sentiment scores as features
│   └── options_greeks.py         # Black-Scholes Greeks (Δ, Γ, Θ, Vega)
│
├── models/                       # ML models
│   ├── lgbm_model.py             # LightGBM with walk-forward training
│   ├── catboost_meta_labeler.py  # CatBoost meta-labeler + Conformal Prediction
│   ├── conformal_predictor.py    # Inductive Conformal Prediction (ICP)
│   ├── tft_model.py              # Temporal Fusion Transformer (PyTorch)
│   ├── tabnet_model.py           # TabNet attention-based model
│   └── xgboost_model.py          # XGBoost gradient boosting
│
├── nlp/                          # Natural Language Processing
│   ├── finbert_sentiment.py      # FinBERT financial sentiment analysis
│   ├── knowledge_graph.py        # Entity-relationship knowledge graph
│   ├── news_velocity.py          # News flow velocity tracker
│   └── vector_db.py              # Milvus vector database for semantic search
│
├── regime/                       # Market regime detection
│   └── hmm_regime.py             # Gaussian HMM regime classifier
│
├── risk/                         # Risk management
│   ├── position_sizing.py        # Volatility-scaled Kelly criterion
│   ├── portfolio_risk.py         # Portfolio-level risk metrics (VaR, CVaR)
│   ├── dpo_layer.py              # Differentiable Portfolio Optimization
│   └── stop_loss.py              # Dynamic trailing stop-loss engine
│
├── signals/                      # Signal generation
│   ├── ensemble.py               # Regime-conditional weighted ensemble + Platt scaling
│   └── filters.py                # Regime, volatility, and confidence filters
│
├── valuation/                    # Fundamental valuation models
│   ├── dcf_model.py              # Discounted Cash Flow model
│   ├── ddm_model.py              # Dividend Discount Model
│   ├── relative_valuation.py     # Comparable company analysis
│   └── triangulation_engine.py   # Multi-model valuation triangulation
│
├── monitoring/                   # Observability
│   ├── dashboard.py              # Streamlit real-time dashboard
│   ├── drift_detector.py         # Feature & model drift detection
│   └── grafana_exporter.py       # Prometheus metrics exporter
│
├── scripts/                      # Runnable scripts
│   ├── run_full_backtest.py      # End-to-end backtest runner
│   └── run_cpcv_backtest.py      # CPCV backtest runner
│
├── tests/                        # Test suite
│   ├── test_features/            # Feature engineering tests
│   ├── test_signals/             # Ensemble scoring tests
│   ├── test_risk/                # Position sizing tests
│   └── test_integration/         # Pipeline integration tests
│
├── config.yaml                   # System configuration
├── docker-compose.yml            # Full stack orchestration
├── Dockerfile                    # Multi-stage build (builder + runtime)
├── prometheus.yml                # Prometheus scrape config
├── requirements.txt              # Python dependencies
├── run_simulation.py             # Standalone simulation runner
└── .env.example                  # Environment variable template
```

---

## 🔧 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Languages** | Python 3.11+, Rust (PyO3/Maturin) |
| **ML/DL** | LightGBM, CatBoost, XGBoost, PyTorch, TabNet, Transformers (FinBERT) |
| **Math** | Conformal Prediction, Platt Scaling, Black-Scholes, Almgren-Chriss, Kelly Criterion |
| **Data** | Pandas, NumPy, SciPy, Scikit-learn |
| **Broker API** | Zerodha Kite Connect (WebSocket + REST) |
| **Databases** | TimescaleDB (PostgreSQL), Redis, Milvus (Vector DB) |
| **Monitoring** | Prometheus, Grafana, Streamlit |
| **Infra** | Docker, Docker Compose, MLflow |
| **Testing** | Pytest |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Rust toolchain (for core engine compilation)
- Docker & Docker Compose (recommended)
- Zerodha Kite API credentials (for live trading)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/intraday-quant-system.git
cd intraday-quant-system

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Launch the full stack
docker compose up --build -d
```

This starts all services:
| Service | Port | Description |
|---------|------|-------------|
| Pipeline | 8000 | Main trading pipeline + Prometheus metrics |
| Streamlit | 8501 | Real-time trading dashboard |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Monitoring dashboards |
| Redis | Internal | Pub/sub message broker |
| TimescaleDB | Internal | Time-series database |

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Build Rust core engine (optional, for tick processing)
cd core_engine
maturin develop --release
cd ..

# Configure
cp .env.example .env
# Edit .env and config.yaml

# Run the pipeline
python -m deployment.pipeline_runner
```

---

## ⚙ Configuration

All system parameters are centralized in [`config.yaml`](config.yaml):

```yaml
trading:
  universe: ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
  max_capital: 1000000.0
  trading_start: "09:15"
  trading_end: "15:15"

risk:
  max_risk_per_trade: 0.02       # 2% of capital
  max_open_positions: 5
  daily_loss_limit: 0.03         # 3% — triggers kill switch
  max_drawdown_limit: 0.10       # 10% — pause for review

transaction_costs:                # NSE equity intraday
  brokerage_pct: 0.0003
  stt_sell_pct: 0.00025
  estimated_slippage_pct: 0.0005
```

Environment-specific secrets go in `.env`:

```env
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
TIMESCALEDB_URL=postgresql://user:password@localhost:5432/quant_db
REDIS_URL=redis://localhost:6379/0
MAX_CAPITAL=1000000
```

---

## 💻 Usage

### News-Driven Scanner (Standalone)

```bash
# Scan mode — fetches live RSS news, identifies sectors, generates trade recommendations
python app.py --mode scan

# Live mode — simulated WebSocket tick consumption with real-time signal generation
python app.py --mode live --universe RELIANCE,TCS,HDFCBANK --interval 1
```

### Full Pipeline

```bash
# Run the multi-process inference pipeline (requires Redis)
python -m deployment.pipeline_runner

# Run simulation with mock tick injection
python run_simulation.py
```

---

## 📈 Backtesting

```bash
# Full walk-forward backtest with realistic execution simulation
python -m scripts.run_full_backtest

# Combinatorial Purged Cross-Validation (CPCV) backtest
python -m scripts.run_cpcv_backtest
```

**Backtest realism features:**
- ✅ VWAP-based fill prices (not close/open)
- ✅ Almgren-Chriss asymmetric market impact model
- ✅ Queue position simulation based on order size vs. book depth
- ✅ Tiered slippage by market cap (large: 3bps, mid: 10bps, small: 30bps)
- ✅ Full NSE transaction cost model (brokerage, STT, exchange charges, GST, SEBI, stamp duty)

---

## 🧪 Testing

```bash
# Run full test suite
pytest

# Run specific test modules
pytest tests/test_features/       # Feature engineering
pytest tests/test_signals/        # Ensemble scoring & calibration
pytest tests/test_risk/           # Position sizing & risk limits
pytest tests/test_integration/    # Pipeline initialization
```

---

## 📊 Monitoring

| Tool | URL | Purpose |
|------|-----|---------|
| **Streamlit** | `http://localhost:8501` | Real-time P&L, positions, signals dashboard |
| **Grafana** | `http://localhost:3000` | System metrics, latency histograms, alerting |
| **Prometheus** | `http://localhost:9090` | Raw metrics and query interface |

**Built-in drift detection** monitors feature distributions and model prediction drift, triggering alerts when statistical thresholds are breached.

---

## ⚠ Disclaimer

> **This software is for educational and research purposes only.**
>
> - This is **not financial advice**. Trading in financial markets involves substantial risk of loss.
> - Past performance in backtests does **not guarantee** future results.
> - The authors are **not responsible** for any financial losses incurred through the use of this system.
> - Always conduct your own due diligence before deploying any trading system with real capital.
> - Ensure compliance with all applicable SEBI regulations and exchange rules before live deployment.

---

<div align="center">

**Built with ❤️ for the Indian markets**

</div>
]]>
