import time

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

weather_df = pd.read_csv(r'C:\Users\Saarthak\Desktop\datasets\Delhi2017-2024Load\weather_may_june.csv')
weather_df['Datetime'] = pd.to_datetime(weather_df['Datetime'], format="%d/%m/%Y %H:%M")
weather_df.set_index(weather_df['Datetime'], inplace=True)

# Assuming 'weather_df' is your weather DataFrame
weather_resampled = weather_df.resample('15min').ffill()  # Resample to 15 minutes
weather_resampled = weather_resampled.asfreq('15min')  # Set frequency explicitly
weather_resampled.index.freq = '15min'
weather_df.sort_index()
# weather_resampled = weather_resampled.loc['2024-06-01 00:00:00':'2024-06-26 23:45:00']
last_row = weather_resampled.iloc[-1]
new_timestamps = ['2024-06-26 23:15:00', '2024-06-26 23:30:00', '2024-06-26 23:45:00']
new_rows = pd.DataFrame([last_row.values] * len(new_timestamps), columns=weather_resampled.columns,
                        index=new_timestamps)

# Concatenate the new rows with the existing DataFrame
weather_resampled = pd.concat([weather_resampled, new_rows])
# weather_resampled.to_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\weather.xlsx')
print(weather_resampled.to_string())

# Loading Data into Dataframe
df = pd.read_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\processed.xlsx')
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d/%m/%Y %H:%M")
# print(df.dtypes)
df.set_index(df['Datetime'], inplace=True)
df = df.asfreq('15min')  # Set frequency explicitly
df.index.freq = '15min'  # Ensure frequency is set correctly
df.sort_index()

forecast_start = pd.to_datetime('2024-06-9 00:00:00')

# print(df)
for i in range(8, 26):
    train_start = df.index[0] + pd.Timedelta(days=2588 + i)
    train_end = df.index[0] + pd.Timedelta(days=2618 + i) - pd.Timedelta(minutes=15)
    train_data = df.loc[train_start:train_end]
    test_start = train_end + pd.Timedelta(minutes=15)
    test_data = df.loc[test_start - pd.Timedelta(hours=17) + pd.Timedelta(minutes=15):test_start + pd.Timedelta(
        days=1) - pd.Timedelta(minutes=15)]

    train = pd.DataFrame(train_data)
    test = pd.DataFrame(test_data)

    train['wind'] = weather_resampled['w']
    test['wind'] = weather_resampled['w']
    test['wind'] = test['wind'].ffill()

    # Assuming your DataFrame is named 'df'
    rows_to_remove = 4 * 17
    train = train.iloc[:-rows_to_remove + 1]
    # print(train)
    # print(test)

    model = SARIMAX(endog=train['Load'],
                    exog=train['wind'],
                    order=(2, 0, 0),
                    seasonal_order=(1, 1, 0, 96))

    # Fit the model
    model_fit = model.fit(disp=False)
    # Print the model summary
    # model_fit = joblib.load('model.joblib')  # Forecast future values
    # print(model_fit.summary())
    time.sleep(5)
    forecast_periods = 163
    forecast = model_fit.get_forecast(steps=forecast_periods, exog=test['wind'])
    # forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    # print(forecast_mean)
    forecast_df = pd.DataFrame({'Forecast': forecast_mean})
    # print(forecast_df)
    # forecast_df.to_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\forecast.xlsx')
    # Plot the forecast
    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index, df['Load'], label='train')
    # plt.plot(test.index, test['Load'], label='test')
    # plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
    # plt.xlabel("Date")
    # plt.ylabel("Load")
    # plt.legend()
    # plt.show()

    # joblib.dump(model_fit, 'model.joblib')
    # forecast_df = pd.DataFrame({'Forecast': forecast_mean})

    forecast_df = forecast_df.loc[forecast_start:(forecast_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15))]
    test = test.loc[forecast_start:(forecast_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15))]
    print(forecast_df)
    print(test)
    mape = mean_absolute_percentage_error(test['Load'], forecast_df['Forecast'])
    print('mape :', mape * 100)

    forecast_start = forecast_start + pd.Timedelta(days=1)
    time.sleep(5)
