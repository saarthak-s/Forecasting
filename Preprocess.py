# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# #
# # df = pd.read_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\Brpl_Load.xlsx')
# # start_date = pd.Timestamp("2017-04-01 00:00:00")
# # end_date = pd.Timestamp("2024-06-26 23:45:00")
# # date_range = pd.date_range(start=start_date, end=end_date, freq="15min")
# #
# # # Add the datetime column to your existing DataFrame
# # df["Datetime"] = date_range
# #
# # df.to_excel(r'C:\Users\Saarthak\Desktop\datasets\Delhi 2017-2024 Load\pqrocessed.xlsx')
#
# import pandas as pd
#
# for j in range(25,26):
#     # Load the first file
#     first_file_path = f'C:/Users/Saarthak/Desktop/datasets/Delhi 2017-2024 Load/model/IntraDay_values/result_j{j+1}_i1.xlsx'
#
#     first_df = pd.read_excel(first_file_path)
#     date_col = first_df.columns[0]
#     first_df.set_index(date_col, inplace=True)
#
#     # Create a dictionary to store dataframes for each file
#     dataframes_by_index = {}
#     common_time_range = first_df.index
#
#     # Iterate over the remaining files (i2, i3, ..., i14)
#     for i in range(2, 16):
#         file_path = f"C:/Users/Saarthak/Desktop/datasets/Delhi 2017-2024 Load/model/IntraDay_values/result_j{j+1}_i{i}.xlsx"
#         df = pd.read_excel(file_path)
#
#         # Assume the first column contains the date information
#         date_col = df.columns[0]
#         df.set_index(date_col, inplace=True)
#
#         # Extract data only within the common time range
#         df = df.reindex(common_time_range)
#         print(df)
#         # Store the dataframe in the dictionary
#         dataframes_by_index[i] = df
#
#     # Append columns to the first file based on date index
#     for i in range(2, 16):
#         actual_col = f'actual_i{i}'
#         forecast_col = f'forecast_i{i}'
#         first_df[actual_col] = dataframes_by_index[i]['Test']
#         first_df[forecast_col] = dataframes_by_index[i]['Forecast']
#         print(dataframes_by_index[i]['Forecast'])
#         print(first_df)
#
#
#     first_df.reset_index(inplace=True)
#     # Save the updated data back to the first file
#     first_df.to_excel(first_file_path, engine='openpyxl')
#
#     print("Data appended successfully based on date index (using positional assumption)!")
