#Retreiving the Alberta Trade Data
import pandas as pd
import numpy as np
import pyodbc


conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=c-goa-sql-10593;'
    'DATABASE=JET_TRDAT_UAT;'
    'Trusted_Connection=yes;'
)

Country_codes = pd.read_sql("SELECT * FROM dbo.countryCodesDescriptionStatCanUNComTradeUSCensusBureau", conn)
# Country codes are not unique, so we need to drop duplicates
Country_codes = Country_codes.loc[:, ['ctyCode', 'UNComTradeCtyId']]
Country_codes = Country_codes.drop_duplicates(subset=['ctyCode'], keep='first')
# Notice all differennt UNComTradeCtyId are included in the furhter processing dat
# SO Here for each ctyCode I pick the first available UNComTradeCtyId
# And drop the rest to prevfent double accounting

for year in range(2013, 2024):
    print(year)
    exports = pd.read_sql(f"SELECT * FROM dbo.statCanExportDataMonthly{year}", conn)
    exports = exports.groupby(['hs6Code', 'ctyCode', 'provCode'])[['Value', 'Quantity']].sum().reset_index()
    exports['Year'] = year
    exports = exports.loc[: ,[ 'Year', 'hs6Code', 'ctyCode', 'provCode', 'Value', 'Quantity']]
    exports.rename(columns={'ctyCode': 'Country', 'provCode': 'Province'}, inplace=True)
    Alberta = exports.loc[exports['Province'] == 'AB'].reset_index(drop=True)
    Alberta['Province'] = 'Alberta'
    Alberta = pd.merge(Alberta, Country_codes, how='left', left_on='Country', right_on='ctyCode')
    Alberta.drop(columns=['Country', 'ctyCode'], inplace=True)
    Alberta.rename(columns={'Year': 't','Province': 'i', 'UNComTradeCtyId':'j', 'hs6Code': 'k', 'Value':'v', 'Quantity':'q'}, inplace=True)

    Canada = exports.groupby(['hs6Code', 'Country'])[['Value', 'Quantity']].sum().reset_index()
    Canada['Province'] = 'Canada'
    Canada['Year'] = year
    Canada = pd.merge(Canada, Country_codes[['UNComTradeCtyId', 'ctyCode']], how='left', left_on='Country', right_on='ctyCode')
    Canada.drop(columns=['Country', 'ctyCode'], inplace=True)
    Canada.rename(columns={'Year': 't','Province': 'i', 'UNComTradeCtyId':'j', 'hs6Code': 'k', 'Value':'v', 'Quantity':'q'}, inplace=True)

    
    CEPII = pd.read_csv(f'../TradeHorizonScan/src/Pre-processing/data/CEPII/BACI_HS12_Y{year}_V202501.csv')
    # Transform the Value and Quantity columns to match the CEPII calculations

    # CEPII values are in thousands of dollars, so we need to divide by 1000
    Alberta['v'] = Alberta['v'] / 1000
    Canada['v'] = Canada['v'] / 1000    




    Alberta = Alberta.loc[:, CEPII.columns]
    Canada = Canada.loc[:, CEPII.columns]
    CEPII = pd.concat([CEPII, Alberta], ignore_index=True)
