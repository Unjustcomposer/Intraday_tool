# Intraday_tool

**Autonomous intraday quantitative trading framework for the NSE.**

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)
![Platform: NSE](https://img.shields.io/badge/Platform-NSE%20India-orange)

---

## Overview

An institutional-grade intraday trading system for India's National Stock Exchange, built entirely in Python. It runs as **two decoupled processes** — an ML signal engine that generates trade calls offline, and a separate execution engine that monitors Level 2 order book depth in real time — so a broker WebSocket failure never crashes model training and vice versa. Built for quantitative traders and ML engineers who want a production-grade reference implementation, not a toy backtest.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESS 1: ML SIGNAL ENGINE                  │
│                    (scripts/generate_calls.py)                  │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────┐   │
│  │ STAGE 1  │   │   STAGE 2    │   │       STAGE 3         │   │
│  │ Screener │──▶│ ML Ensemble  │──▶│   11-Filter Cascade   │   │
│  │          │   │              │   │                       │   │
│  │ F&O scan │   │ LightGBM    │   │ Meta-labeler gate     │   │
│  │ ≥₹50     │   │ TabNet      │   │ VIX kill switch       │   │
│  │ ADV≥200K │   │ CatBoost    │   │ R:R ≥ 2.0            │   │
│  │          │   │ Meta-Labeler│   │ Min margin 0.75%      │   │
│  └──────────┘   └──────────────┘   │ Max SL 1.5%          │   │
│                                     │ Portfolio exposure    │   │
│                                     └───────────┬───────────┘   │
└─────────────────────────────────────────────────┼───────────────┘
                                                  │
                                          approved_calls.csv
                                                  │
               ═══════════════════════════════════╪═══════════════
                        DECOUPLING BOUNDARY       │
               ═══════════════════════════════════╪═══════════════
                                                  │
┌─────────────────────────────────────────────────┼───────────────┐
│                  PROCESS 2: EXECUTION ENGINE                    │
│                  (scripts/execution_engine.py)                  │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │   STAGE 4    │   │  OFI Check   │   │  Fyers Broker    │    │
│  │ L2 WebSocket │──▶│  Iceberg     │──▶│  Order Placement │    │
│  │ Depth Monitor│   │  Detection   │   │  INTRADAY mode   │    │
│  └──────────────┘   └──────────────┘   └──────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Value |
|---|---|
| Out-of-sample AUC (LightGBM) | **0.556** |
| Signal filter rate | **~99%** of raw signals rejected |
| Position sizing | Half-Kelly, capped at 10% per trade |
| Max drawdown limit | 10% → full shutdown |
| Daily / Weekly loss limit | 3% / 6% |
| VIX kill switches | Caution at 25, full block at 30 |
| Broker integration | Fyers V3, headless TOTP login |
| Calibration method | Isotonic → Spearman ρ check → Platt fallback |

---

## Tech Stack

| Category | Tools |
|---|---|
| ML Models | LightGBM, PyTorch TabNet, CatBoost (meta-labeler), Scikit-learn |
| Features | VPIN, Kyle's Lambda, Amihud illiquidity, OFI, Garman-Klass vol, Hurst exponent |
| Execution | Fyers V3 WebSocket, Level 2 depth, OFI-based iceberg detection |
| Data | Pandas, NumPy, yfinance, Parquet storage |
| Risk | Half-Kelly sizing, India VIX filters, conformal prediction thresholds |
| Regime | Gaussian Mixture Model (3-state: quiet / trending / volatile + crisis override) |

---

## Installation

```bash
# Clone
git clone https://github.com/Unjustcomposer/Intraday_tool.git
cd Intraday_tool/intraday_quant_system

# Virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Dependencies
pip install -r requirements.txt
```

Configure `.env` in the project root:

```ini
FYERS_CLIENT_ID=YOUR_APP_ID-200
FYERS_SECRET_KEY=YOUR_SECRET
FYERS_ACCESS_TOKEN=               # Generated daily via auth.py
```

Run the system:

```bash
# Step 1: Generate ML signals
python -m scripts.generate_calls --symbols RELIANCE.NS HDFCBANK.NS --days 60

# Step 2: Launch execution engine (separate process)
python scripts/execution_engine.py
```

---

## Project Structure

```
intraday_quant_system/
├── scripts/
│   ├── generate_calls.py        # ML signal engine (Process 1)
│   ├── execution_engine.py      # Execution daemon  (Process 2)
│   ├── simulate_historical.py   # Historical walk-forward simulation
│   └── run_full_backtest.py     # Full backtest with Monte Carlo
├── models/
│   ├── lgbm_model.py            # LightGBM with purged CV + isotonic calibration
│   ├── tabnet_model.py          # PyTorch TabNet (decorrelated ensemble member)
│   └── catboost_meta_labeler.py # Meta-labeler with conformal prediction
├── signals/
│   ├── ensemble.py              # Regime-conditional weighted scoring
│   └── call_generator.py        # Trade call generation + 11-filter cascade
├── features/
│   ├── feature_store.py         # 22+ microstructure & volatility features
│   ├── microstructure.py        # VPIN, trade sign correlation
│   └── volatility_features.py   # ATR, Garman-Klass, realized vol
├── data/
│   ├── market_data.py           # yfinance + Fyers data engine
│   └── fyers_client.py          # Broker client, WebSocket, L2 cache, OFI
├── regime/
│   └── hmm_regime.py            # GMM-based regime detection (3-state + crisis)
├── deployment/
│   └── config.py                # Transaction costs, risk limits, timing (Pydantic)
└── .env                         # Credentials (gitignored)
```

---

## What's Next

- **Options pricing:** Black-Scholes integration for NSE F&O, Greeks-based hedging
- **Event-driven architecture:** Redis pub/sub replacing CSV handoff between processes
- **Live paper trading:** Real-time P&L dashboard with fill tracking and slippage measurement

---

## Disclaimer

This software is for educational and research purposes only. Do not risk money you cannot afford to lose. Use at your own risk. The author assumes no responsibility for trading results.

---

Built by **Harshit Khandelwal** · [LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN) · [Read the full breakdown on Medium](https://medium.com/@YOUR_HANDLE)
