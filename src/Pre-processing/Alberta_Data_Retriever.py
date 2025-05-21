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


def Alberta_to_CEPII(Alberta, Canada , CEPII):
    # This function will return the Alberta data adjusted for the CEPII methodology
    # By calculating Alberta(Statscan) Q an  V using the proportion of Canada_CEPII/StatsCan_Canada Q and V
    Canada_CEPII = CEPII.loc[CEPII['i'] == 124].reset_index(drop=True) #Exporter is Canada
    Country_map = pd.read_sql("SELECT * FROM dbo.countryCodesDescriptionStatCanUNComTradeUSCensusBureau", conn)
    Country_map = Country_map.loc[:, ['ctyCode', 'UNComTradeCtyId']]
    Canada_CEPII = pd.merge(Canada_CEPII, Country_map, how='left', left_on='j', right_on='UNComTradeCtyId')
    Canada_CEPII.drop(columns=['UNComTradeCtyId'], inplace=True)
    Canada_CEPII = Canada_CEPII.groupby(['t', 'i', 'ctyCode', 'k'])[['v', 'q']].sum().reset_index()
    Canada_CEPII.rename(columns={'v': 'canada_cepii_v', 'q': 'canada_cepii_q'}, inplace=True)
    Alberta_df = pd.merge(Alberta, Country_map, how='left', left_on='j', right_on='UNComTradeCtyId')
    Alberta_df.drop(columns=['UNComTradeCtyId'], inplace=True)
    Alberta_df.rename(columns={'v': 'alberta_v', 'q': 'alberta_q'}, inplace=True)
    assert Alberta_df.duplicated(subset=['ctyCode', 'k']).sum() == 0, "There are duplicates in the Alberta data"
    Canada_df = pd.merge(Canada, Country_map, how='left', left_on='j', right_on='UNComTradeCtyId')
    Canada_df.drop(columns=['UNComTradeCtyId'], inplace=True)
    Canada_df.rename(columns={'v': 'canada_v', 'q': 'canada_q'}, inplace=True)
    assert Canada_df.duplicated(subset=['ctyCode', 'k']).sum() == 0, "There are duplicates in the Alberta data"
    Alberta_df = pd.merge(Alberta_df, Canada_df[['canada_v', 'canada_q', 'ctyCode', 'k']], how='left', on=['ctyCode', 'k'])
    Alberta_df = pd.merge(Alberta_df, Canada_CEPII[['canada_cepii_v', 'canada_cepii_q', 'ctyCode', 'k']], how='left', on=['ctyCode', 'k'])

    Alberta_df['adjusted_v'] = Alberta_df['alberta_v'] * (Alberta_df['canada_v']/Alberta_df['canada_cepii_v'])
    Alberta_df['adjusted_q'] = Alberta_df['alberta_q'] * (Alberta_df['canada_q']/Alberta_df['canada_cepii_q'])
    Alberta_df['adjusted_v'] = Alberta_df['adjusted_v'].fillna(Alberta_df['alberta_v'])
    Alberta_df['adjusted_q'] = Alberta_df['adjusted_q'].fillna(Alberta_df['alberta_q'])
    Alberta_df = Alberta_df.loc[:, ['t', 'i','j', 'k', 'adjusted_v', 'adjusted_q']]
    Alberta_df.rename(columns={'adjusted_v': 'v', 'adjusted_q': 'q'}, inplace=True)
    return Alberta_df

for year in range(2013, 2024):
    print(year)
    exports = pd.read_sql(f"SELECT * FROM dbo.statCanExportDataMonthly{year}", conn)
    exports.ctyCode = exports.ctyCode.replace("TW", "CN") #Modifying Trades with Taiwan to trade with China to keep it consistent with the ComTrade and World Bank data
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
    

    # CEPII values are in thousands of dollars, so we need to divide by 1000
    Alberta['v'] = Alberta['v'] / 1000
    Canada['v'] = Canada['v'] / 1000
    # Transform the Value and Quantity columns to match the CEPII calculations


    Alberta = Alberta_to_CEPII(Alberta, Canada, CEPII)
    Alberta = Alberta.loc[:, CEPII.columns]
    CEPII2 = pd.concat([CEPII, Alberta], ignore_index=True)
    print(f'for the year {year} the number of rows in the CEPII data increased from {CEPII.shape[0]:,} to {CEPII2.shape[0]:,}')
    CEPII2.to_csv(f'../TradeHorizonScan/src/Pre-processing/data/CEPII/BACI_HS12_Y{year}_V202501_alberta.csv', index=False)


