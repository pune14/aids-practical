import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
flights = sns.load_dataset('flights')
flights['date'] = pd.to_datetime(flights['year'].astype(str) + '-' + flights['month'].astype(str) + '-01')
ts = flights.set_index('date')['passengers'].asfreq('MS')
print("Observations:", len(ts), " From", ts.index[0].date(), "to", ts.index[-1].date())
ts.plot(figsize=(8, 4))
plt.title("Monthly Airline Passengers")
plt.ylabel("Passengers (thousands)")
plt.grid(True)
plt.show()
train, test = ts[:-24], ts[-24:]
model = ExponentialSmoothing(train, trend='add', seasonal='mul',seasonal_periods=12).fit()
forecast = model.forecast(24)
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = np.mean(np.abs((test - forecast) / test)) * 100
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")
plt.figure(figsize=(8, 4))
plt.plot(train, label="Train")
plt.plot(test, label="Actual (Test)")
plt.plot(forecast, label="Forecast", linestyle="--")
plt.title("Holt-Winters Forecast")
plt.legend()
plt.grid(True)
plt.show()
