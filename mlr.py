import time
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# Load weather data
weather_df = pd.read_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\weather may-june2024.xlsx')
weather_df['Datetime'] = pd.to_datetime(weather_df['Datetime'], format="%d/%m/%Y %H:%M")
weather_df.set_index(weather_df['Datetime'], inplace=True)

# Resample to 15 minutes and handle missing values
weather_resampled = weather_df.resample('15min').ffill()
weather_resampled = weather_resampled.asfreq('15min')
weather_resampled.index.freq = '15min'
weather_df.sort_index()

# Add additional rows
last_row = weather_resampled.iloc[-1]
new_timestamps = ['2024-06-26 23:15:00', '2024-06-26 23:30:00', '2024-06-26 23:45:00']
new_rows = pd.DataFrame([last_row.values] * len(new_timestamps), columns=weather_resampled.columns,
                        index=new_timestamps)
weather_resampled = pd.concat([weather_resampled, new_rows])
# print(weather_resampled.to_string())

# Load the main dataset
df = pd.read_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\processed.xlsx')
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d/%m/%Y %H:%M")
df.set_index(df['Datetime'], inplace=True)
df = df.asfreq('15min')
df.index.freq = '15min'
df.sort_index()

forecast_start = pd.to_datetime('2024-06-09 00:00:00')

train_start = df.index[0] + pd.Timedelta(days=2588)
train_end = df.index[0] + pd.Timedelta(days=2618) - pd.Timedelta(minutes=15)
train_data = df.loc[train_start:train_end]
test_start = train_end + pd.Timedelta(minutes=15)
test_data = df.loc[test_start - pd.Timedelta(hours=17) + pd.Timedelta(minutes=15):test_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)]

train = pd.DataFrame(train_data)
test = pd.DataFrame(test_data)

train[['wind', 'temp', 'preci']] = weather_resampled[['w', 't2m', 'tp']]
test[['wind', 'temp', 'preci']] = weather_resampled[['w', 't2m', 'tp']]
test[['wind', 'temp', 'preci']] = test[['wind', 'temp', 'preci']].ffill()
rows_to_remove = 4 * 17
train = train.iloc[:-rows_to_remove + 1]
train['Load_diff'] = train['Load'].diff(periods=96).dropna()
test['Load_diff'] = test['Load'].diff(periods=96).dropna()

print(train.tail().to_string())
print(test.tail().to_string())

model = LinearRegression()

model.fit(train['Load_diff'], train['Load'])


# forecast_periods = 163
forecast = model.predict(test['Load_diff'])
print(forecast)
forecast_df = pd.DataFrame({'Forecast': forecast}, index=test.index)
forecast_df = forecast_df.loc[forecast_start:(forecast_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15))]
test = test.loc[forecast_start:(forecast_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15))]

print(forecast_df)
print(test)

mape = mean_absolute_percentage_error(test['Load'], forecast_df['Forecast'])
print('MAPE:', mape * 100)

