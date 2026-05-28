import yfinance as yf
symbol = "^NSEI"
ticker = yf.Ticker(symbol)
data = ticker.history(period="5d", interval="5m")
print(f"Data for {symbol}:")
print(data.tail())
