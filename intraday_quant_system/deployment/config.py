import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally


class TransactionCosts(BaseModel):
    """Indian market transaction cost model (NSE equity intraday)"""
    brokerage_pct: float = 0.0003
    stt_sell_pct: float = 0.00025
    exchange_txn_pct: float = 0.0000345
    gst_pct: float = 0.18
    sebi_turnover_pct: float = 0.000001
    stamp_duty_buy_pct: float = 0.00003
    estimated_slippage_pct: float = 0.0002
    
    def total_cost_buy(self, turnover: float) -> float:
        """Total cost for a buy trade"""
        brokerage = min(turnover * self.brokerage_pct, 20.0)  # Zerodha ₹20 cap
        exchange = turnover * self.exchange_txn_pct
        gst = (brokerage + exchange) * self.gst_pct
        sebi = turnover * self.sebi_turnover_pct
        stamp = turnover * self.stamp_duty_buy_pct
        slippage = turnover * self.estimated_slippage_pct
        return brokerage + exchange + gst + sebi + stamp + slippage
    
    def total_cost_sell(self, turnover: float) -> float:
        """Total cost for a sell trade"""
        brokerage = min(turnover * self.brokerage_pct, 20.0)
        stt = turnover * self.stt_sell_pct
        exchange = turnover * self.exchange_txn_pct
        gst = (brokerage + exchange) * self.gst_pct
        sebi = turnover * self.sebi_turnover_pct
        slippage = turnover * self.estimated_slippage_pct
        return brokerage + stt + exchange + gst + sebi + slippage
    
    def total_round_trip_pct(self) -> float:
        """Approximate total round-trip cost as percentage"""
        return (
            self.brokerage_pct * 2
            + self.stt_sell_pct
            + self.exchange_txn_pct * 2
            + (self.brokerage_pct * 2 + self.exchange_txn_pct * 2) * self.gst_pct
            + self.sebi_turnover_pct * 2
            + self.stamp_duty_buy_pct
            + self.estimated_slippage_pct * 2
        )


class SuccessMetrics(BaseModel):
    """Minimum thresholds for production go-live"""
    min_sharpe_ratio: float = 1.0
    max_drawdown_pct: float = 0.10
    min_win_rate: float = 0.50
    min_profit_factor: float = 1.3
    min_walk_forward_splits_passing: float = 0.70


class RiskConfig(BaseModel):
    """Risk management parameters"""
    max_risk_per_trade: float = 0.02
    max_open_positions: int = 5
    max_sector_exposure: float = 0.25
    max_portfolio_exposure: float = 0.70
    min_cash_reserve: float = 0.30
    daily_loss_limit: float = 0.03
    weekly_loss_limit: float = 0.06
    max_drawdown_limit: float = 0.10
    vix_cutoff: float = 25.0


class IntradayTiming(BaseModel):
    """Intraday session parameters for NSE"""
    bars_per_day: int = 75
    bars_per_year: int = 18900
    no_new_trades_after: str = "14:30"
    begin_closing: str = "15:10"
    hard_exit: str = "15:15"


class LGBMConfig(BaseModel):
    num_leaves: int = 63
    learning_rate: float = 0.03
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    min_child_samples: int = 100


class CatBoostConfig(BaseModel):
    iterations: int = 800
    learning_rate: float = 0.05
    depth: int = 5
    early_stopping_rounds: int = 50
    l2_leaf_reg: float = 10.0


class TransformerConfig(BaseModel):
    embed_dim: int = 64
    nhead: int = 4
    num_layers: int = 3
    seq_len: int = 120
    epochs: int = 100
    patience: int = 10


class ModelsConfig(BaseModel):
    lgbm: LGBMConfig = Field(default_factory=LGBMConfig)
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)


class QuantConfig(BaseModel):
    zerodha_api_key: str = Field(default_factory=lambda: os.getenv("KITE_API_KEY", ""))
    zerodha_api_secret: str = Field(default_factory=lambda: os.getenv("KITE_API_SECRET", ""))
    timescaledb_url: str = Field(default_factory=lambda: os.getenv("TIMESCALEDB_URL", ""))
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    universe: List[str] = Field(default_factory=list)
    target_volatility: float = 0.15
    max_capital: float = 1000000.0
    trading_start: str = "09:15"
    trading_end: str = "15:15"
    retraining_frequency: str = "weekly"
    
    # Sub-configs
    transaction_costs: TransactionCosts = Field(default_factory=TransactionCosts)
    success_metrics: SuccessMetrics = Field(default_factory=SuccessMetrics)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    intraday: IntradayTiming = Field(default_factory=IntradayTiming)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "QuantConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found at {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}
            
        config_dict = {}
        
        # Extract trading config
        if 'trading' in yaml_data:
            trading = yaml_data['trading']
            for key in ['universe', 'target_volatility', 'max_capital', 'trading_start', 'trading_end', 'retraining_frequency']:
                if key in trading:
                    config_dict[key] = trading[key]
                    
        # Extract zerodha config
        if 'zerodha' in yaml_data:
            z = yaml_data['zerodha']
            if not os.getenv("KITE_API_KEY") and 'api_key' in z:
                val = z['api_key']
                if not val.startswith('${'):
                    config_dict['zerodha_api_key'] = val
            if not os.getenv("KITE_API_SECRET") and 'api_secret' in z:
                val = z['api_secret']
                if not val.startswith('${'):
                    config_dict['zerodha_api_secret'] = val
                    
        # Database fallback to yaml if not in env
        if 'database' in yaml_data:
            db = yaml_data['database']
            if not os.getenv("TIMESCALEDB_URL") and 'timescaledb_url' in db:
                val = db['timescaledb_url']
                if not val.startswith('${'):
                    config_dict['timescaledb_url'] = val
            if not os.getenv("REDIS_URL") and 'redis_url' in db:
                val = db['redis_url']
                if not val.startswith('${'):
                    config_dict['redis_url'] = val
                    
        # Transaction costs
        if 'transaction_costs' in yaml_data:
            config_dict['transaction_costs'] = TransactionCosts(**yaml_data['transaction_costs'])
        
        # Success metrics
        if 'success_metrics' in yaml_data:
            config_dict['success_metrics'] = SuccessMetrics(**yaml_data['success_metrics'])
        
        # Risk config
        if 'risk' in yaml_data:
            config_dict['risk'] = RiskConfig(**yaml_data['risk'])
        
        # Intraday timing
        if 'intraday' in yaml_data:
            config_dict['intraday'] = IntradayTiming(**yaml_data['intraday'])
            
        # Models config
        if 'models' in yaml_data:
            models_data = yaml_data['models']
            lgbm_cfg = LGBMConfig(**models_data.get('lgbm', {}))
            cat_cfg = CatBoostConfig(**models_data.get('catboost', {}))
            trans_cfg = TransformerConfig(**models_data.get('transformer', {}))
            config_dict['models'] = ModelsConfig(lgbm=lgbm_cfg, catboost=cat_cfg, transformer=trans_cfg)
                
        return cls(**config_dict)


# Global instance
def get_config(yaml_path: str = "config.yaml") -> QuantConfig:
    if os.path.exists(yaml_path):
        return QuantConfig.load_from_yaml(yaml_path)
    return QuantConfig()
