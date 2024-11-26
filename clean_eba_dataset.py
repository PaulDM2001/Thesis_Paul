import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import pickle

with open('dataset_EBA.csv', 'r', encoding='utf-8', errors='replace') as file:
    df_EBA_raw = pd.read_csv(file)

EU_COUNTRIES_list = ["AT", "BE", "CY", "DE", "DK", "ES", "FI", "FR", "GB", "GR", "HU", "IE", "IT", "LU", "MT", "NL", "NO", "PL", "PT", "SE", "SI"]

## Extract necessary informatiton
total_equity = df_EBA_raw[(df_EBA_raw['INFORMATION_CODE'] == 30021) & (df_EBA_raw['INFO_DATE_CODE'] == 20101231.0)].iloc[:, 7:].astype(float).T  # Get equity value 31-12-2010
total_assets = df_EBA_raw[(df_EBA_raw['INFORMATION_CODE'] == 30029) & (df_EBA_raw['INFO_DATE_CODE'] == 20101231.0)].iloc[:, 7:].astype(float).T
total_interbank_assets = df_EBA_raw[(df_EBA_raw['INFORMATION_CODE'] == 33010) & (df_EBA_raw['C_COUNTRY_CODE'] == 'TO')].iloc[:, 7:].astype(float).T
total_EU_interbank_assets = df_EBA_raw[(df_EBA_raw['INFORMATION_CODE'] == 33010) & (df_EBA_raw['C_COUNTRY_CODE'].isin(EU_COUNTRIES_list))].iloc[:, 7:].astype(float).sum(axis=0).to_frame()
total_NONEU_interbank_assets = (total_interbank_assets.iloc[:, 0] - total_EU_interbank_assets.iloc[:, 0]).to_frame()
total_interbank_liabilities = total_interbank_assets
total_external_assets = (total_assets.iloc[:, 0] - total_interbank_liabilities.iloc[:, 0]).to_frame()
total_external_liabilities = (total_assets.iloc[:, 0] - total_interbank_liabilities.iloc[:, 0] - total_equity.iloc[:, 0]).to_frame() 

## Create new dataframe 
df_cleaned_dict = {'Total Equity': total_equity.iloc[:, 0], 'Total Assets': total_assets.iloc[:, 0], 'Total EU interbank assets': total_EU_interbank_assets.iloc[:, 0], 'Total nonEU interbank assets': total_NONEU_interbank_assets.iloc[:, 0], 
                   'Total Interbank Assets': total_interbank_assets.iloc[:, 0], 'Total Interbank Liabilities': total_interbank_liabilities.iloc[:, 0], 'Total External Assets': total_external_assets.iloc[:, 0], 
                   'Total External Liabilities': total_external_liabilities.iloc[:, 0]}
df_EBA_cleaned = pd.DataFrame(df_cleaned_dict)

## Data checks
# print(df_EBA_cleaned.isna().sum())                                          # check for NA values
rows_negative = df_EBA_cleaned[(df_EBA_cleaned < 0).any(axis=1)].index      # check for negative values 
# print(df_EBA_cleaned[(df_EBA_cleaned < 0).any(axis=1)])                     # three banks have either 0 TTA in the report or negative External Assets / External Liabilities (DE029, LU045, SI058)
df_EBA_cleaned.drop(rows_negative, inplace=True)                            # remove these banks
# print(df_EBA_cleaned)

## Export DF
df_EBA_cleaned.to_csv('dataset_EBA_cleaned.csv')