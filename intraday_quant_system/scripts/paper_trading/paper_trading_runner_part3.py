def run_paper_trading(self, days: int = 10) -> bool:
        """
        Run paper trading for specified number of days.
        """
        logger.info(f"Starting paper trading for {days} days...")
        
        if not self.models_loaded:
            logger.error("Models not loaded. Call load_models() or train_models() first.")
            return False
        
        # Fetch initial data for feature computation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        all_data = {}
        for symbol in self.symbols:
            logger.info(f"Fetching initial data for {symbol}...")
            df = self.market_data.fetch_fyers_historical_data(symbol, start_date, end_date)
            if not df.empty:
                all_data[symbol] = df
        
        if not all_data:
            logger.error("No initial data fetched")
            return False
        
        # Compute initial features
        all_features = {}
        for symbol, df in all_data.items():
            features_df = self.feature_store.compute_all(symbol, df)
            all_features[symbol] = features_df
        
        # Fit regime detector on combined data
        combined_features = pd.concat(all_features.values())
        self.regime_detector.fit(combined_features)
        
        # Main trading loop
        logger.info("Starting paper trading loop...")
        self.state.start_date = datetime.now().strftime("%Y-%m-%d")
        self.state.end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        
        try:
            for day in range(days):
                current_date = datetime.now() + timedelta(days=day)
                
                # Skip weekends
                if current_date.weekday() >= 5:
                    logger.info(f"Skipping weekend: {current_date.strftime('%Y-%m-%d')}")
                    continue
                
                logger.info(f"Trading day {day+1}/{days}: {current_date.strftime('%Y-%m-%d')}")
                
                # Run single trading day
                success = self._run_trading_day(current_date)
                
                if not success:
                    logger.error(f"Trading day failed for {current_date}")
                    break
                
                # Save daily state
                self._save_daily_state(current_date)
                
                # Small delay between days
                if day < days - 1:
                    time.sleep(1)
            
            self.state.end_date = datetime.now().strftime("%Y-%m-%d")
            logger.info("Paper trading completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Paper trading failed: {e}")
            return False
    
    def _run_trading_day(self, current_date: datetime) -> bool:
        """Run a single trading day."""
        try:
            # Market hours: 9:15 - 15:15
            market_open = dt_time(9, 15)
            market_close = dt_time(15, 15)
            
            logger.info(f"Market open at {market_open}")
            
            # Pre-market screening (9:00 - 9:15)
            logger.info("Running pre-market screener...")
            if not self._run_pre_market_screening():
                logger.warning("Pre-market screening failed")
            
            # Main trading loop (9:15 - 15:15)
            # In paper trading, we simulate the day using historical data
            return self._simulate_trading_day()
            
        except Exception as e:
            logger.error(f"Trading day failed: {e}")
            return False
    
    def _run_pre_market_screening(self) -> bool:
        """Run pre-market screener to find stocks in play."""
        try:
            # Use screener to find active symbols
            self.symbols = self.screener.scan_pre_market()
            logger.info(f"Pre-market screener found {len(self.symbols)} symbols: {self.symbols}")
            return True
        except Exception as e:
            logger.error(f"Pre-market screening failed: {e}")
            return False
    
    def _simulate_trading_day(self) -> bool:
        """
        Simulate a full trading day using historical data.
        In paper trading, we use the day's historical data to simulate trades.
        """
        try:
            # For each symbol, simulate the day's trading
            for symbol in self.symbols:
                # Get day's data (in real trading, this would be real-time)
                # For paper trading, we use the day's historical data
                pass
            
            # Process signals through order manager
            # This is where the actual trading logic runs
            self._process_trading_signals()
            
            return True
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False
    
    def _run_pre_market_screening(self) -> bool:
        """Run pre-market screener to find stocks in play."""
        try:
            # Use screener to find active symbols
            self.symbols = self.screener.scan_pre_market()
            logger.info(f"Pre-market screener found {len(self.symbols)} symbols: {self.symbols}")
            return True
        except Exception as e:
            logger.error(f"Pre-market screening failed: {e}")
            return False
    
    def _simulate_trading_day(self) -> bool:
        """
        Simulate a full trading day using historical data.
        In paper trading, we use the day's historical data to simulate trades.
        """
        try:
            # For each symbol, simulate the day's trading
            for symbol in self.symbols:
                # Get day's data (in real trading, this would be real-time)
                # For paper trading, we use the day's historical data
                pass
            
            # Process signals through order manager
            # This is where the actual trading logic runs
            self._process_trading_signals()
            
            return True
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False
    
    def _run_pre_market_screening(self) -> bool:
        """Run pre-market screener to find stocks in play."""
        try:
            # Use screener to find active symbols
            self.symbols = self.screener.scan_pre_market()
            logger.info(f"Pre-market screener found {len(self.symbols)} symbols: {self.symbols}")
            return True
        except Exception as e:
            logger.error(f"Pre-market screening failed: {e}")
            return False
    
    def _simulate_trading_day(self) -> bool:
        """
        Simulate a full trading day using historical data.
        In paper trading, we use the day's historical data to simulate trades.
        """
        try:
            # For each symbol, simulate the day's trading
            for symbol in self.symbols:
                # Get day's data (in real trading, this would be real-time)
                # For paper trading, we use the day's historical data
                pass
            
            # Process signals through order manager
            # This is where the actual trading logic runs
            self._process_trading_signals()
            
            return True
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False
    
    def _process_trading_signals(self):
        """Process trading signals through the full pipeline."""
        try:
            # Get current market data
            current_prices = {}
            for symbol in self.symbols:
                # In paper trading, use mock prices
                current_prices[symbol] = 2500.0 + np.random.normal(0, 10)
            
            # Get features for current state
            # In real trading, this would be computed from live data
            features_data = {}
            
            # Generate signals
            signals = self._generate_signals()
            
            # Process through order manager
            if signals:
                self.order_manager.process_signals(
                    signals_df=signals,
                    current_prices=current_prices,
                    market_data=None,
                    features_data=None
                )
            
            # Manage open positions
            self.order_manager.manage_open_positions(current_prices)
            
        except Exception as e:
            logger.error(f"Signal processing failed: {e}")
    
    def _generate_signals(self):
        """Generate trading signals from models."""
        # This would use the trained models to generate signals
        # For paper trading, return empty for now
        return None