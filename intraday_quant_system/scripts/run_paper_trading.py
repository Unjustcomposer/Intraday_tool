"""
Paper Trading Orchestrator
==========================
Master script to run the daily paper trading loop. 
Intended to be run at market open (e.g., 09:15 IST).
- Step 1: Generates calls via ML Pipeline
- Step 2: Spawns the execution engine in paper mode
- Step 3: Waits for market close
- Step 4: Runs validation and Go/No-Go framework
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("PaperRunner")

def run():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("\n" + "═"*60)
    print(" 🚀 INTRADAY QUANT SYSTEM: PAPER TRADING RUNNER")
    print("═"*60)
    
    # --- Step 1: Generate Calls ---
    logger.info("Step 1: Running ML Signal Generator...")
    gen_script = os.path.join(base_dir, "scripts", "generate_calls.py")
    
    try:
        # Run synchronous and wait
        subprocess.run([sys.executable, gen_script], check=True)
    except subprocess.CalledProcessError:
        logger.error("Signal generation failed! Aborting paper trading for today.")
        sys.exit(1)
        
    pending_csv = os.path.join(base_dir, "data", "pending_orders.csv")
    if not os.path.exists(pending_csv):
        logger.info("No pending orders generated today. Exiting.")
        sys.exit(0)
        
    # --- Step 2: Spawn Execution Engine ---
    logger.info("Step 2: Launching Execution Engine in PAPER mode...")
    exec_script = os.path.join(base_dir, "scripts", "execution_engine.py")
    
    # We spawn this as a subprocess so it can monitor the websocket
    exec_process = subprocess.Popen([sys.executable, exec_script, "--env", "paper"])
    
    # --- Step 3: Wait until Market Close ---
    # Market closes at 15:30 IST. 
    # For testing, we will just run it for a set duration if a flag is passed, 
    # or wait until the actual time.
    
    close_time = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    logger.info(f"Engine running. Will auto-shutdown at {close_time.strftime('%H:%M:%S')}.")
    
    try:
        while True:
            now = datetime.now()
            if now >= close_time:
                logger.info("Market Closed. Shutting down Execution Engine.")
                exec_process.terminate()
                break
            time.sleep(60) # check every minute
    except KeyboardInterrupt:
        logger.info("Manual interrupt received. Shutting down Execution Engine.")
        exec_process.terminate()
        
    # Wait for graceful shutdown
    exec_process.wait()
    
    # --- Step 4: Validation & Reporting ---
    logger.info("Step 3: Generating End-of-Day Paper Report...")
    report_script = os.path.join(base_dir, "scripts", "generate_paper_report.py")
    subprocess.run([sys.executable, report_script])
    
    logger.info("Step 4: Evaluating Go/No-Go Framework...")
    gonogo_script = os.path.join(base_dir, "deployment", "go_nogo_framework.py")
    subprocess.run([sys.executable, gonogo_script])
    
    print("\n🏁 Paper Trading cycle complete for today.")

if __name__ == "__main__":
    run()
