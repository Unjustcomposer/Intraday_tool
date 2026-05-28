# NSE/BSE Institutional-Grade Intraday Quant Trading System

This is a fully functional, modular Python application replicating an institutional-grade intraday quantitative trading system for Indian equities (NSE/BSE).

## Architecture
- **Data Ingestion**: Zerodha Kite API, TimescaleDB, Redis
- **Feature Engineering**: Flow, Trend, Volatility, Sentiment, Market Context
- **Regime Detection**: GaussianHMM, Autoencoder
- **Alpha Models**: LightGBM, CatBoost (Meta-Labeling), PyTorch Temporal Transformer
- **NLP**: FinBERT Sentiment Analysis
- **Execution**: Smart Execution Engine with constraints (3:15 PM Hard Exit)
- **Monitoring**: Streamlit, Grafana, Prometheus

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Configure settings in `config.yaml` and `.env`.
4. Run the pipeline: `python -m deployment.pipeline_runner`
