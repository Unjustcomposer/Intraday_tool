import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Quant System Dashboard", layout="wide")

def main():
    st.title("Intraday Quant System Dashboard")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Live P&L + Equity Curve",
        "Open Positions + Risk Exposure",
        "Signal Log",
        "Feature Drift Alerts",
        "Regime State",
        "Daily Kill Switch"
    ])
    
    # Mock Data Generation
    if 'equity_curve' not in st.session_state:
        st.session_state.equity_curve = [1000000]
        
    st.session_state.equity_curve.append(st.session_state.equity_curve[-1] * (1 + np.random.normal(0.0001, 0.001)))
    
    if page == "Live P&L + Equity Curve":
        st.header("Live P&L")
        current_equity = st.session_state.equity_curve[-1]
        st.metric("Total Equity", f"₹{current_equity:,.2f}", f"{(current_equity/1000000 - 1):.2%}")
        st.line_chart(st.session_state.equity_curve)
        
    elif page == "Open Positions + Risk Exposure":
        st.header("Open Positions")
        # Mock positions
        df = pd.DataFrame({
            'Symbol': ['RELIANCE', 'TCS'],
            'Side': ['BUY', 'SELL'],
            'Qty': [100, 50],
            'Entry': [2500.5, 3800.0],
            'Current': [2510.0, 3790.0],
            'P&L': [950.0, 500.0]
        })
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Risk Exposure")
        col1, col2 = st.columns(2)
        col1.metric("Gross Exposure", "15.2%")
        col2.metric("Net Exposure", "4.1%")
        
    elif page == "Signal Log":
        st.header("Signal Log")
        st.write("Recent generated signals")
        
    elif page == "Feature Drift Alerts":
        st.header("Feature Drift Detection")
        st.warning("No significant drift detected.")
        
    elif page == "Regime State":
        st.header("Current Market Regime")
        st.info("Current Regime: High Volatility Trend")
        
    elif page == "Daily Kill Switch":
        st.header("Risk Kill Switch Status")
        st.success("All systems green. No kill switch triggered.")

if __name__ == "__main__":
    main()
