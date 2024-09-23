import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, image as mpimg
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import joblib
from statsmodels.tsa.stl.mstl import MSTL

#
# # Loading Data into Dataframe
df = pd.read_csv(r'C:\Users\Saarthak\Desktop\datasets\Delhi2017-2024Load\processed2.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
# print(df.dtypes)
df.set_index(df['Datetime'], inplace=True)
df = df.asfreq('15min')  # Set frequency explicitly
df.index.freq = '15min'  # Ensure frequency is set correctly
df.sort_index()


#
# Perform ADF test
def adf_test(timeseries):
    print('Results of ADF Test:')
    dftest = adfuller(timeseries.dropna())
    output = pd.Series(dftest[0:3], index=['Test Statistic', 'p-value', '#Lags Used'])

    print(output)


# adf_test(df['Load'])
# Output: Non stationary time series


def kpsstest(timeseries):
    print('Results of KPSS Test:')
    dftest = kpss(timeseries.dropna())
    kpss_output = pd.Series(dftest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    print(kpss_output)


#
#
# # kpsstest(df['Load'])
# # Output: Non stationary time series

df['Load_diff'] = df['Load'].diff(periods=96).dropna()  # First Order Differencing
plt.plot(df['Datetime'], df['Load'])
plt.show()
train_start = df.index[0] + pd.Timedelta(days=2560)
train_end = train_start + pd.Timedelta(days=60) - pd.Timedelta(minutes=15)
train = df.loc[train_start:train_end]
plt.plot(train['Datetime'], train['Load'])
plt.show()
# First Order Differencing

# print(train_data)
# plt.scatter(df['Datetime'], df['Load_diff'])
adf_test(train['Load'].dropna())
kpsstest(train['Load'].dropna())
# time.sleep(15)
adf_test(train['Load_diff'].dropna())
kpsstest(train['Load_diff'].dropna())
# Output: Stationary time series

# Plotting ACF and PACF plots
# fig, ax = plt.subplots(2, 1, figsize=(10, 8))
# plot_acf(train['Load_diff'].dropna(), lags=200, ax=ax[0])
plot_pacf(train['Load_diff'].dropna(), lags=200)
plt.show()
# Observation : PACF cut off at lag 3 ,ACF Gradually decreasing
#
# plt.rc('figure',figsize=(12,8))
# plt.rc('font',size=15)
# result = seasonal_decompose(df['Load'],model='additive',period=35040)
# result.plot()
# plt.rc('figure',figsize=(12,6))
# plt.rc('font',size=15)
# fig, ax = plt.subplots()
# x = result.resid.index
# y = result.resid.values
# ax.plot_date(x, y, color='black',linestyle='--')
# fig.autofmt_xdate()
# plt.show()
#
# train_end = df.index[-1] - pd.Timedelta(days=2)
# train_start = df.index[0] + pd.Timedelta(days=2618)
# train_data = df.loc[train_start:train_end]
# test_data = df.loc[train_end + pd.Timedelta(minutes=15):]
# train = pd.DataFrame(train_data)
# test = pd.DataFrame(test_data)
# train['Load_diff'] = train['Load'] - train['Load'].shift(96)  # First Order Differencing
#
# print(kpss(train['Load_diff'].dropna()))
# adf_test(train['Load_diff'].dropna())
#
# plt.plot(train['Load'])
# plt.show()
#

# # decomposition = seasonal_decompose(train['Load'], model='additive',period= 15)  # 96 periods for daily seasonality in 15-minute intervals
# # trend = decomposition.trend
# # seasonal = decomposition.seasonal
# # residual = decomposition.resid
# #
# # Plot components
# plt.figure(figsize=(12, 8))
# plt.subplot(411)
# plt.plot(train['Load'], label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal, label='Seasonal')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residual')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
#
#
# # Create subplots
# fig, axes = plt.subplots(2, 1, figsize=(12, 8))
#
# # ACF plot
# plot_acf(train['Load_diff'].dropna(), lags=500, ax=axes[0])
# axes[0].set_title('ACF Plot')
#
# # PACF plot
# plot_pacf(train['Load_diff'].dropna(), lags=500, ax=axes[1])
# axes[1].set_title('PACF Plot')
# #
# # # Show plots
# plt.tight_layout()
# plt.show()
#
# monthly_segments = df.resample('ME').mean()
#
# # Example of how to access data for a specific month (e.g., July)
# january_data = monthly_segments[monthly_segments.index.month == 1]
# february_data = monthly_segments[monthly_segments.index.month == 2]
# march_data = monthly_segments[monthly_segments.index.month == 3]
# april_data = monthly_segments[monthly_segments.index.month == 4]
# may_data = monthly_segments[monthly_segments.index.month == 5]
# june_data = monthly_segments[monthly_segments.index.month == 6]
# july_data = monthly_segments[monthly_segments.index.month == 7]
# august_data = monthly_segments[monthly_segments.index.month == 8]
# september_data = monthly_segments[monthly_segments.index.month == 9]
# october_data = monthly_segments[monthly_segments.index.month == 10]
# november_data = monthly_segments[monthly_segments.index.month == 11]
# december_data = monthly_segments[monthly_segments.index.month == 12]
#
# plt.plot(january_data.index, january_data['Load'], label='January')
# plt.plot(february_data.index, february_data['Load'], label='February')
# plt.plot(march_data.index, march_data['Load'], label='March')
# plt.plot(april_data.index, april_data['Load'], label='April')
# plt.plot(may_data.index, may_data['Load'], label='May')
# plt.plot(june_data.index, june_data['Load'], label='June')
# plt.plot(july_data.index, july_data['Load'], label='July')
# plt.plot(august_data.index, august_data['Load'], label='August')
# plt.plot(september_data.index, september_data['Load'], label='September')
# plt.plot(october_data.index, october_data['Load'], label='October')
# plt.plot(november_data.index, november_data['Load'], label='November')
# plt.plot(december_data.index, december_data['Load'], label='December')
#
# plt.xlabel('Datetime')
# plt.ylabel('Load')
# plt.title('Load vs Datetime for Different Months')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.show()

df_2024 = df.loc['2024-01-01':'2024-06-26']
sundays = df_2024[df_2024.index.day_name().isin(['Saturday', 'Sunday'])]
print(sundays)
# sundays.to_csv(r"C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\Saturday_sunday_load.csv")
# Filter data for other days (excluding Sundays)
other_days = df_2024[df_2024.index.day_name() != 'Sunday']

#
plt.plot(df_2024.index, df_2024['Load'], label='All Days', color='gray', alpha=0.7)

# Plot load values for Sundays
plt.plot(sundays.index, sundays['Load'], label='Sundays', color='orange', linewidth=2)

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Load Value')
plt.title('Load Values: All Days vs. Sundays')
plt.legend()

# Show the plot
plt.show()
