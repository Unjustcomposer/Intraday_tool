def load_models(self) -> bool:
        """Load pre-trained models from disk."""
        try:
            model_dir = "./data/models"
            if not os.path.exists(model_dir):
                logger.warning(f"Model directory {model_dir} not found, will train new models")
                return self.train_models()
            
            # Load LGBM
            lgbm_path = os.path.join(model_dir, "lgbm_latest.txt")
            if os.path.exists(lgbm_path):
                self.lgbm_model.load(lgbm_path)
                logger.info(f"Loaded LGBM from {lgbm_path}")
            
            # Load Meta-Labeler
            meta_path = os.path.join(model_dir, "meta_latest.cbm")
            if os.path.exists(meta_path):
                self.meta_labeler.load(meta_path)
                logger.info(f"Loaded Meta-Labeler from {meta_path}")
            
            # Load Regime Detector
            regime_path = os.path.join(model_dir, "regime_latest.pkl")
            if os.path.exists(regime_path):
                self.regime_detector.load(regime_path)
                logger.info(f"Loaded Regime Detector from {regime_path}")
            
            self.models_loaded = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def train_models(self) -> bool:
        """Train all models on historical data."""
        try:
            logger.info("Training models on historical data...")
            
            # Fetch historical data for all symbols
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            all_data = {}
            for symbol in self.symbols:
                logger.info(f"Fetching data for {symbol}...")
                df = self.market_data.fetch_fyers_historical_data(symbol, start_date, end_date)
                if not df.empty:
                    all_data[symbol] = df
            
            if not all_data:
                logger.error("No historical data fetched")
                return False
            
            # Compute features for all symbols
            all_features = {}
            for symbol, df in all_data.items():
                logger.info(f"Computing features for {symbol}...")
                features_df = self.feature_store.compute_all(symbol, df)
                all_features[symbol] = features_df
            
            # Train models for each symbol
            for symbol, features_df in all_features.items():
                logger.info(f"Training models for {symbol}...")
                
                # Generate labels
                labels = LGBMAlphaModel.make_labels(features_df)
                features_df['label'] = labels
                features_df = features_df.dropna()
                
                if len(features_df) < 200:
                    logger.warning(f"Insufficient data for {symbol}: {len(features_df)} samples")
                    continue
                
                feature_cols = self.feature_store.get_feature_columns()
                feature_cols = [c for c in feature_cols if c in features_df.columns]
                
                X = features_df[feature_cols]
                y = features_df['label']
                
                # Train LGBM
                self.lgbm_model.train(X, y)
                
                # Train Meta-Labeler
                # Split: 70% primary, 30% meta
                split_idx = int(len(X) * 0.7)
                X_primary = X.iloc[:split_idx]
                y_primary = y.iloc[:split_idx]
                X_meta = X.iloc[split_idx:]
                y_meta = y.iloc[split_idx:]
                
                primary_preds = self.lgbm_model.predict(X_meta)
                y_meta_outcome = (primary_preds == y_meta).astype(int)
                
                self.meta_labeler.train(primary_preds, X_meta, y_meta_outcome)
                
                # Fit regime detector
                self.regime_detector.fit(features_df)
            
            # Save models
            os.makedirs("./data/models", exist_ok=True)
            self.lgbm_model.save("./data/models/lgbm_latest.txt")
            self.meta_labeler.save("./data/models/meta_latest.cbm")
            self.regime_detector.save("./data/models/regime_latest.pkl")
            
            self.models_loaded = True
            logger.info("Models trained and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            return False