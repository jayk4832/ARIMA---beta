import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime as dt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

def GetStockData(ticker_name, period, start_date, end_date):
  tickerData = yf.Ticker(ticker_name)
  df = tickerData.history(period=period, start=start_date, end=end_date)
  return df

x = dt.datetime.now()

nvda_df = GetStockData("NVDA", "1d", "2016-01-01", x)

nvda_df = nvda_df[['Close']].copy()

#Augmented Dickey Fuller Test
dftest = adfuller(nvda_df["Close"], autolag="AIC")

#If data is non-stationary then the p-value >= 0.05.
dfourput = pd.Series(dftest[0:4], index=["Test Stats", "p-value", "# Lags", "# of obs"])
for key, value in dftest[4].items():
  dfourput[f"Critical Value ({key})"] = value
print(dfourput)

nvda_df["First_Difference"] = nvda_df["Close"].diff()
nvda_df["Second_Difference"] = nvda_df["First_Difference"].diff()

fig = make_subplots(rows=2, cols=1)

for idx, d in enumerate(["First_Difference", "Second_Difference"]):
  fig.add_trace(
      go.Scatter(
          name = d,
          x = nvda_df.index,
          y = nvda_df[d]
      ),
      row = idx + 1,
      col = 1
      )
  
fig.update_layout(
    title="Differenced Stock Prices",
    xaxis_title="Date",
    yaxis_title="Price"
)

fig.show()

acf(nvda_df["First_Difference"].dropna());

model = pm.auto_arima(nvda_df["Close"], start_p=1, start_q=1, test="adf", max_p=3, max_q=3, m=1, d=None, seasonal=False,
                      start_P=0, D=0, trace=True, error_action="ignore", suppress_warnings=True, stepwise=True)

print(model.summary())

import matplotlib.pyplot as plt

train = nvda_df.Close[:1000]
test  = nvda_df.Close[1000:]

# Updated forecast method
model = ARIMA(train, order=(1, 1, 0))
fit_model = model.fit()

# Get forecast and confidence intervals
forecast_results = fit_model.get_forecast(steps=203, alpha=0.05)
fc = forecast_results.predicted_mean
conf = forecast_results.conf_int()

fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf.iloc[:, 0], index=test.index)
upper_series = pd.Series(conf.iloc[:, 1], index=test.index)

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()