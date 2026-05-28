import threading
import time
import json
import redis
import pandas as pd
import numpy as np
from datetime import datetime
from data.bar_aggregator import BarAggregator
from deployment.pipeline_runner import PipelineRunner
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Simulation")

class SimulationRunner:
    """
    Simulation of the V3 Multi-Process Pipeline.
    
    1. Starts Process B (PipelineRunner) in a thread.
    2. Starts Process A (BarAggregator) in main thread.
    3. Feeds mock ticks into BarAggregator.
    4. Observes signal generation and execution.
    """
    def __init__(self, redis_url="redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.r = redis.from_url(self.redis_url)
        
        # Initialize PipelineRunner (Process B)
        self.pipeline = PipelineRunner()
        self.pipeline.is_running = True # Force running state
        
        # Initialize Aggregator (Process A)
        self.aggregator = BarAggregator(redis_url=self.redis_url, interval_mins=1) # 1-min bars for fast simulation
        
    def start_inference_process(self):
        """Process B loop"""
        logger.info("Thread: Process B started")
        try:
            self.pipeline.main_loop()
        except Exception as e:
            logger.error(f"Process B error: {e}")

    def run_simulation(self, duration_mins=5):
        """Main simulation loop"""
        # Start Process B thread
        inference_thread = threading.Thread(target=self.start_inference_process, daemon=True)
        inference_thread.start()
        
        logger.info("Process A: Starting mock tick injection")
        
        symbol = "RELIANCE"
        start_time = datetime.now()
        price = 2500.0
        
        # Inject ticks for 'duration_mins' simulation minutes
        for i in range(duration_mins * 60): # 60 seconds per simulated minute
            tick = {
                'symbol': symbol,
                'price': price + np.random.normal(0, 1),
                'volume': np.random.randint(10, 100),
                'timestamp': datetime.now()
            }
            
            self.aggregator.on_tick(tick)
            
            # If it's the end of a minute, the aggregator will publish to Redis
            # which will trigger Process B
            
            time.sleep(0.1) # Accelerated simulation: 1 second = 100ms
            
        logger.info("Simulation complete. Shutting down.")
        self.pipeline.is_running = False
        self.aggregator.force_finalize_all()

if __name__ == "__main__":
    # Note: Requires a local Redis server running
    try:
        sim = SimulationRunner()
        sim.run_simulation(duration_mins=2)
    except Exception as e:
        logger.error(f"Simulation failed to start. Ensure Redis is running: {e}")
