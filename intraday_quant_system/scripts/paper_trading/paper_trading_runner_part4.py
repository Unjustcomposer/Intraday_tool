def _save_daily_state(self, current_date: datetime):
        """Save daily state to disk."""
        try:
            state_file = f"./data/paper_trading/state_{current_date.strftime('%Y%m%d')}.json"
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2, default=str)
            logger.info(f"Daily state saved to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save daily state: {e}")
    
    def generate_validation_report(self) -> dict:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        # Calculate final metrics
        trades = self.state.trades
        if not trades:
            return {"error": "No trades executed"}
        
        # Basic metrics
        total_trades = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning) / total_trades if total_trades > 0 else 0
        total_pnl = sum(t.pnl for t in trades)
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        profit_factor = abs(sum(t.pnl for t in winning) / sum(t.pnl for t in losing)) if losing else float('inf')
        
        # Calculate Sharpe
        returns = [t.pnl for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max drawdown
        equity_curve = np.cumsum([0] + [t.pnl for t in trades])
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Monte Carlo stress test
        mc = MonteCarloStressTester(n_simulations=5000)
        mc_result = mc.run(pd.DataFrame({'return': [t.pnl for t in trades]}))
        
        # DSR
        returns = np.array([t.pnl for t in trades])
        dsr = calculate_dsr(
            np.array([t.pnl for t in trades]),
            n_trials=1000,
            variance_of_sharpes=0.5,
            annualization_factor=252
        )
        
        report = {
            "period": {
                "start": self.state.start_date,
                "end": self.state.end_date,
                "days": (datetime.strptime(self.state.end_date, "%Y-%m-%d") - 
                         datetime.strptime(self.state.start_date, "%Y-%m-%d")).days
            },
            "capital": {
                "initial": self.state.initial_capital,
                "final": self.state.current_capital,
                "total_pnl": self.state.total_pnl,
                "return_pct": (self.state.current_capital - self.state.initial_capital) / self.state.initial_capital * 100
            },
            "trading": {
                "total_trades": self.state.total_trades,
                "winning_trades": self.state.winning_trades,
                "losing_trades": self.state.losing_trades,
                "win_rate": self.state.win_rate,
                "profit_factor": self.state.profit_factor,
                "avg_win": self.state.avg_win,
                "avg_loss": self.state.avg_loss,
                "max_consecutive_losses": self.state.max_consecutive_losses,
            },
            "risk": {
                "max_drawdown": self.state.max_drawdown,
                "max_drawdown_pct": self.state.max_drawdown_pct,
                "sharpe_ratio": self.state.sharpe_ratio,
                "win_rate": self.state.win_rate,
            },
            "monte_carlo": mc_result,
            "deflated_sharpe": dsr.dsr,
            "go_no_go": self._make_go_no_go_decision()
        }
        
        return report
    
    def _make_go_no_go_decision(self) -> dict:
        """Make go/no-go decision based on validation criteria."""
        criteria = {
            "sharpe_ratio": {
                "value": self.state.sharpe_ratio,
                "threshold": 1.0,
                "pass": self.state.sharpe_ratio >= 1.0
            },
            "win_rate": {
                "value": self.state.win_rate,
                "threshold": 0.50,
                "pass": self.state.win_rate >= 0.50
            },
            "profit_factor": {
                "value": self.state.profit_factor,
                "threshold": 1.3,
                "pass": self.state.profit_factor >= 1.3
            },
            "max_drawdown": {
                "value": self.state.max_drawdown_pct,
                "threshold": 0.10,
                "pass": self.state.max_drawdown_pct <= 0.10
            },
            "min_trades": {
                "value": self.state.total_trades,
                "threshold": 30,
                "pass": self.state.total_trades >= 30
            }
        }
        
        all_pass = all(c["pass"] for c in criteria.values())
        
        return {
            "decision": "GO" if all_pass else "NO-GO",
            "criteria": criteria,
            "reason": "All criteria met" if all_pass else "Some criteria not met"
        }
    
    def save_report(self, report: dict, filepath: str = None):
        """Save validation report to file."""
        if filepath is None:
            filepath = f"./data/paper_trading/report_{datetime.now().strftime('%Y%m%d')}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {filepath}")
    
    def run_validation(self, days: int = 10) -> dict:
        """Run complete paper trading validation."""
        logger.info("Starting paper trading validation...")
        
        # Initialize
        if not self.initialize():
            return {"error": "Initialization failed"}
        
        # Load or train models
        if not self.load_models():
            if not self.train_models():
                return {"error": "Failed to load or train models"}
        
        # Run paper trading
        if not self.run_paper_trading(days=10):
            return {"error": "Paper trading failed"}
        
        # Generate report
        report = self.generate_validation_report()
        
        # Save report
        self.save_report(report)
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Validation Runner")
    parser.add_argument("--symbols", nargs="+", default=["RELIANCE", "HDFCBANK", "TCS", "INFY"], 
                        help="Symbols to trade")
    parser.add_argument("--days", type=int, default=10, help="Number of days to run")
    parser.add_argument("--capital", type=float, default=1000000.0, help="Initial capital")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--train", action="store_true", help="Train models before running")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    
    # Create engine
    engine = PaperTradingEngine(
        symbols=args.symbols,
        initial_capital=args.capital,
        config_path=args.config,
        paper_trading=True
    )
    
    if args.train:
        logger.info("Training models...")
        if not engine.train_models():
            logger.error("Model training failed")
            return 1
    
    # Run validation
    report = engine.run_validation(days=args.days)
    
    if "error" in report:
        logger.error(f"Validation failed: {report['error']}")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print("PAPER TRADING VALIDATION REPORT")
    print("="*60)
    print(f"Decision: {report['go_no_go']['decision']}")
    print(f"Reason: {report['go_no_go']['reason']}")
    print(f"Period: {report['period']['start']} to {report['period']['end']}")
    print(f"Total PnL: {report['capital']['total_pnl']:.2f} ({report['capital']['return_pct']:.2f}%)")
    print(f"Total Trades: {report['trading']['total_trades']}")
    print(f"Win Rate: {report['trading']['win_rate']:.2%}")
    print(f"Profit Factor: {report['trading']['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {report['risk']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {report['risk']['max_drawdown_pct']:.2%}")
    print(f"Monte Carlo P(Profit): {report['monte_carlo']['probability_of_profit']:.2%}")
    print(f"DSR: {report['deflated_sharpe']:.4f}")
    print("="*60)
    
    return 0 if report['go_no_go']['decision'] == 'GO' else 1


if __name__ == "__main__":
    sys.exit(main())