import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt

today = dt.date.today()

# stock = input("Which stock would you like to view? (e.g., MSFT, AAPL, TSLA) *Please enter in symbol*: ")
stock = 'MSFT'

ticker = yf.Ticker(stock)
data = ticker.history(period = 'max')
print(data[['Open', 'High', 'Low', 'Close']])

plt.title(f"{stock} Data")
plt.figure(figsize=(10,6))
plt.grid(True)
plt.plot(data.index, data['Close'], color='r', label='close')
plt.plot(data.index, data['Open'], color='b', label='open')
plt.xlabel('Date')
plt.ylabel('Closed Price')
plt.legend()
plt.savefig('plot.png')