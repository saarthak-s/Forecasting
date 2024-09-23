import time
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Loading Data into DataFrame
df = pd.read_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi2017-2024Load\processed.xlsx')
df['Datetime'] = pd.to_datetime(df['Datetime'], format="%d/%m/%Y %H:%M")
df.set_index(df['Datetime'], inplace=True)
df = df.asfreq('15min')  # Set frequency explicitly
df.index.freq = '15min'  # Ensure frequency is set correctly
df.sort_index()

for j in range(1, 9):
    mape_values = []
    for i in range(15):
        train_start = df.index[0] + pd.Timedelta(days=2588 + j)
        train_end = train_start + pd.Timedelta(days=30) - pd.Timedelta(minutes=15) + pd.Timedelta(hours=1.5 * i)
        train = df.loc[train_start:train_end]

        test_start = train_end + pd.Timedelta(minutes=15)
        test_end = train_start + pd.Timedelta(days=31) - pd.Timedelta(minutes=15)
        test = df.loc[test_start:test_end]

        print(train)
        print(test)

        auto_model = auto_arima(train['Load'],
                                seasonal=False,
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

        p, d, q = auto_model.order  # Extract non-seasonal parameters

        model = SARIMAX(endog=train['Load'],
                        order=(p, d, q),
                        seasonal_order=(1, 1, 0, 96))

        # Fit the model
        model_fit = model.fit(disp=False, low_memory=True)
        # Print the model summary
        # model_fit = joblib.load('model.joblib')  # Forecast future values
        print(model_fit.summary())
        # time.sleep(5)
        forecast_periods = len(test)
        forecast = model_fit.get_forecast(steps=forecast_periods)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        # print(forecast_mean)
        forecast_df = pd.DataFrame({'Forecast': forecast_mean})
        # print(forecast_df)
        # forecast_df.to_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\forecast.xlsx')
        # Plot the forecast
        # joblib.dump(model_fit, 'model.joblib')
        # forecast_df = pd.DataFrame({'Forecast': forecast_mean})

        forecast_df = forecast_df.loc[train_end + pd.Timedelta(hours=1.75):test_end]
        test = test.loc[train_end + pd.Timedelta(hours=1.75):test_end]
        # print(forecast_df.to_string())
        # print(test.to_string())
        mape = mean_absolute_percentage_error(test['Load'], forecast_df['Forecast'])
        print(f'mape{i}:', mape * 100)
        mape_values.append(mape * 100)

        result_df = pd.DataFrame({'Test': test['Load'], 'Forecast': forecast_df['Forecast']})
        # result_df.to_excel(
        #     f'C:/Users/Saarthak/Desktop/datasets/Delhi 2017-2024 Load/model/IntraDay_values/result_j{j+1}_i{i+1}.xlsx',
        #     index=True)

        # plt.figure(figsize=(12, 6))
        # plt.plot(train.index, train['Load'], label='train')
        # plt.plot(test.index, test['Load'], label='test')
        # plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
        # plt.xlabel("Date")
        # plt.ylabel("Load")
        # plt.legend()
        # plt.savefig(f'image/plot_june{j+1}_i{i+1}.png')  # Save the plot as an image file
        # plt.close()  # Close the figure to free up memory

    # df_mape = pd.DataFrame({'MAPE': mape_values})
    # df_mape.to_excel(f'mape_values{j + 1}.xlsx', index=False)

print("MAPE values stored in 'mape_values.xlsx' files and plots saved as 'plot_j{j+1}_i{i+1}.png'.")
