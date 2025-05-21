import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os



os.chdir(os.path.join(os.getcwd(), 'src', 'Pre-processing', 'data'))
data = pd.read_csv("3- Diversification_Project_Preprocessed.csv", low_memory=False)
data.sort_values(by=['importer', 'exporter', 'hsCode', 'year'], inplace=True)
data.reset_index()
a = data.loc[data.year>=2015]
print(f'Number of Nan values before the transformation: {a.isna().sum().sum():,} cells,  {100* a.isna().sum().sum()/(a.shape[0]*a.shape[1]):.2f} %')



# Calculating the moving average (['importer', 'exporter', 'hsCode'])
# Basically variables that are specific to an importer, exporter and hsCode (and a year obviously)
col = ['value', 'AvgUnitPrice', 'Trade_Complementarity', ]
df = data[['importer', 'exporter', 'hsCode', 'year', 'value', 'AvgUnitPrice', 'Trade_Complementarity']].drop_duplicates()
df = df.sort_values(by=['importer', 'exporter', 'hsCode', 'year'])
df = df.set_index('year').groupby(['importer', 'exporter', 'hsCode'])[col].rolling(window=3, min_periods=1).mean().reset_index()
df.columns = ['importer', 'exporter', 'hsCode','year', 'MA_value', 'MA_AvgUnitPrice', 'MA_Trade_Complementarity']
# Merge the moving average with the original dataframe
data = pd.merge(data, df, on=['year', 'importer', 'exporter', 'hsCode'], how='left')


# Calculating the moving average (['importer', 'exporter'])
# Basically variables that are specific to an importer and exporter (and a year obviously)
col = ['contig', 'dist'] #Obviously there is no need to calculate MA for these variables. Regardless I am doing it as a way to double check my process
df = data[['importer', 'exporter', 'year', 'contig', 'dist']].drop_duplicates()
df = df.sort_values(by=['importer', 'exporter', 'year'])
df = df.set_index('year').groupby(['importer', 'exporter'])[col].rolling(window=3, min_periods=1).mean().reset_index()
df.columns = ['importer', 'exporter', 'year', 'MA_contig', 'MA_dist']
# Merge the moving average with the original dataframe
data = pd.merge(data, df, on=['year', 'importer', 'exporter'], how='left')



# Calculating the moving average (['exporter', 'hsCode])
# Basically variables that are specific to an exporter and hsCode (and a year obviously)
col = ['AvgUnitPriceofExporterToWorld', 'TotalExportofCmdbyPartner', 'Partner_Revealed_Comparative_Advantage']
df = data[['exporter', 'hsCode', 'year', 'AvgUnitPriceofExporterToWorld', 'TotalExportofCmdbyPartner', 'Partner_Revealed_Comparative_Advantage']].drop_duplicates()
df = df.sort_values(by=['exporter', 'hsCode', 'year'])
df = df.set_index('year').groupby(['exporter', 'hsCode'])[col].rolling(window=3, min_periods=1).mean().reset_index()
df.columns = ['exporter', 'hsCode', 'year', 'MA_AvgUnitPriceofExporterToWorld', 'MA_TotalExportofCmdbyPartner', 'MA_Partner_Revealed_Comparative_Advantage']
# Merge the moving average with the original dataframe
data = pd.merge(data, df, on=['year', 'exporter', 'hsCode'], how='left')

# Calculating the moving average (['importer', 'hsCode])
# Basically variables that are specific to an importer and hsCode (and a year obviously)
col = ['AvgUnitPriceofImporterFromWorld', 'TotalImportofCmdbyReporter']
df = data[['importer', 'hsCode', 'year', 'AvgUnitPriceofImporterFromWorld', 'TotalImportofCmdbyReporter']].drop_duplicates()
df = df.sort_values(by=['importer', 'hsCode', 'year'])
df = df.set_index('year').groupby(['importer', 'hsCode'])[col].rolling(window=3, min_periods=1).mean().reset_index()
df.columns = ['importer', 'hsCode', 'year', 'MA_AvgUnitPriceofImporterFromWorld', 'MA_TotalImportofCmdbyReporter']
# Merge the moving average with the original dataframe
data = pd.merge(data, df, on=['year', 'importer', 'hsCode'], how='left')



# Calculating the moving average (['importer'])
# Basically variables that are specific to an importer (and a year obviously)
col = ['Theil_Importer_Concentration', 'GDPPerCapita_importer', 'TariffRatesAllProductsWeigthedAverage_importer', 'GeopoliticalIndex_importer', 'ConsumerPriceIndex_importer']
df = data[['importer', 'year', 'Theil_Importer_Concentration', 'GDPPerCapita_importer', 'TariffRatesAllProductsWeigthedAverage_importer', 'GeopoliticalIndex_importer', 'ConsumerPriceIndex_importer']].drop_duplicates()
df = df.sort_values(by=['importer', 'year'])
df = df.set_index('year').groupby(['importer'])[col].rolling(window=3, min_periods=1).mean().reset_index()
df.columns = ['importer', 'year', 'MA_Theil_Importer_Concentration', 'MA_GDPPerCapita_importer', 'MA_TariffRatesAllProductsWeigthedAverage_importer', 'MA_GeopoliticalIndex_importer', 'MA_ConsumerPriceIndex_importer']
# Merge the moving average with the original dataframe
data = pd.merge(data, df, on=['year', 'importer'], how='left')


# Calculating the moving average (['exporter'])
# Basically variables that are specific to an exporter (and a year obviously)
col = ['Theil_Exporter_Concentration', 'GDPPerCapita_exporter', 'TariffRatesAllProductsWeigthedAverage_exporter', 'GeopoliticalIndex_exporter', 'ConsumerPriceIndex_exporter']
df = data[['exporter', 'year', 'Theil_Exporter_Concentration', 'GDPPerCapita_exporter', 'TariffRatesAllProductsWeigthedAverage_exporter', 'GeopoliticalIndex_exporter', 'ConsumerPriceIndex_exporter']].drop_duplicates()
df = df.sort_values(by=['exporter', 'year'])
df = df.set_index('year').groupby(['exporter'])[col].rolling(window=3, min_periods=1).mean().reset_index()
df.columns = ['exporter', 'year', 'MA_Theil_Exporter_Concentration', 'MA_GDPPerCapita_exporter', 'MA_TariffRatesAllProductsWeigthedAverage_exporter', 'MA_GeopoliticalIndex_exporter', 'MA_ConsumerPriceIndex_exporter']
# Merge the moving average with the original dataframe
data = pd.merge(data, df, on=['year', 'exporter'], how='left')


# Calculating the moving average (['hsCode'])
# Basically variables that are specific to an hsCode (and a year obviously)
col = ['Liberalising', 'Harmful']
df = data[['hsCode', 'year', 'Liberalising', 'Harmful']].drop_duplicates()
df = df.sort_values(by=['hsCode', 'year'])
df = df.set_index('year').groupby(['hsCode'])[col].rolling(window=3, min_periods=1).mean().reset_index()
df.columns = ['hsCode', 'year', 'MA_Liberalising', 'MA_Harmful']
# Merge the moving average with the original dataframe
data = pd.merge(data, df, on=['year', 'hsCode'], how='left')
data.sort_values(by=['importer', 'exporter', 'hsCode', 'year'], inplace=True)



#data = data.loc[data.year>=2015]
a = data.loc[data.year>=2015]
print(f'Number of Nan values before the transformation: {a.isna().sum().sum():,} cells,  {100* a.isna().sum().sum()/(a.shape[0]*a.shape[1]):.2f} %')


# I will replace nan values in UnitPrice columnS with 0s and add a new columns called UnitPriceFlags to indicate the rows that were replaced
data['AvgUnitPriceFlags'] = np.where(data['AvgUnitPrice'].isna(), True, False)
data['AvgUnitPrice'] = data['AvgUnitPrice'].fillna(0)

data['MA_AvgUnitPriceFlags'] = np.where(data['MA_AvgUnitPrice'].isna(), True, False)
data['MA_AvgUnitPrice'] = data['MA_AvgUnitPrice'].fillna(0)

data['AvgUnitPriceofImporterFromWorldFlags'] = np.where(data['AvgUnitPriceofImporterFromWorld'].isna(), True, False)
data['AvgUnitPriceofImporterFromWorld'] = data['AvgUnitPriceofImporterFromWorld'].fillna(0)

data['MA_AvgUnitPriceofImporterFromWorldFlags'] = np.where(data['MA_AvgUnitPriceofImporterFromWorld'].isna(), True, False)
data['MA_AvgUnitPriceofImporterFromWorld'] = data['MA_AvgUnitPriceofImporterFromWorld'].fillna(0)

data['AvgUnitPriceofExporterToWorldFlags'] = np.where(data['AvgUnitPriceofExporterToWorld'].isna(), True, False)
data['AvgUnitPriceofExporterToWorld'] = data['AvgUnitPriceofExporterToWorld'].fillna(0)

data['MA_AvgUnitPriceofExporterToWorldFlags'] = np.where(data['MA_AvgUnitPriceofExporterToWorld'].isna(), True, False)
data['MA_AvgUnitPriceofExporterToWorld'] = data['MA_AvgUnitPriceofExporterToWorld'].fillna(0)


# Including Covid years
# https://www.nm.org/healthbeat/medical-advances/new-therapies-and-drug-trials/covid-19-pandemic-timeline
# Pendemic started in March 2020 and ended in May 2023
# 2020, 2021, and 2022 are flagged as Covid years
data['Covid'] = np.where(data['year'].isin([2020, 2021, 2022]), True, False)


# Rounding the values to 3 decimal places
data = data.round(3)
# Only keeping the after 2015 data
data = data.loc[data.year>=2015]

# Saving the output files
trade_col = ['hsCode', 'year', 'importer', 'exporter', 'value', 'AvgUnitPrice','AvgUnitPriceFlags', 'AvgUnitPriceofImporterFromWorld','AvgUnitPriceofImporterFromWorldFlags', 'TotalImportofCmdbyReporter',
             'AvgUnitPriceofExporterToWorld','AvgUnitPriceofExporterToWorldFlags', 'TotalExportofCmdbyPartner','Trade_Complementarity','Partner_Revealed_Comparative_Advantage', 'Liberalising', 'Harmful', 'Covid']
MA_trade_col = ['hsCode', 'year', 'importer', 'exporter','MA_value', 'MA_AvgUnitPrice', 'MA_AvgUnitPriceFlags', 'MA_AvgUnitPriceofImporterFromWorld', 'MA_AvgUnitPriceofImporterFromWorldFlags', 'MA_TotalImportofCmdbyReporter',
                'MA_AvgUnitPriceofExporterToWorld','MA_AvgUnitPriceofExporterToWorldFlags','MA_AvgUnitPriceofExporterToWorldFlags', 'MA_TotalExportofCmdbyPartner','MA_Trade_Complementarity','MA_Partner_Revealed_Comparative_Advantage', 'MA_Liberalising', 'MA_Harmful', 'Covid']
trade = data[trade_col].drop_duplicates()
MA_trade = data[MA_trade_col].drop_duplicates()
trade.to_csv("output\\Trade.csv", index=False)
MA_trade.to_csv("output\\MA_Trade.csv", index=False)

importer_col = ['importer', 'year', 'Theil_Importer_Concentration', 'GDPPerCapita_importer', 'TariffRatesAllProductsWeigthedAverage_importer', 'GeopoliticalIndex_importer', 'ConsumerPriceIndex_importer']
MA_importer_col = ['importer', 'year', 'MA_Theil_Importer_Concentration', 'MA_GDPPerCapita_importer', 'MA_TariffRatesAllProductsWeigthedAverage_importer', 'MA_GeopoliticalIndex_importer', 'MA_ConsumerPriceIndex_importer']
importer = data[importer_col].drop_duplicates()
MA_importer = data[MA_importer_col].drop_duplicates()
importer.to_csv("output\\Importer.csv", index=False)
MA_importer.to_csv("output\\MA_Importer.csv", index=False)

exporter_col = ['exporter', 'year', 'Theil_Exporter_Concentration', 'GDPPerCapita_exporter', 'TariffRatesAllProductsWeigthedAverage_exporter', 'GeopoliticalIndex_exporter', 'ConsumerPriceIndex_exporter']
MA_exporter_col = ['exporter', 'year', 'MA_Theil_Exporter_Concentration', 'MA_GDPPerCapita_exporter', 'MA_TariffRatesAllProductsWeigthedAverage_exporter', 'MA_GeopoliticalIndex_exporter', 'MA_ConsumerPriceIndex_exporter']
exporter = data[exporter_col].drop_duplicates()
MA_exporter = data[MA_exporter_col].drop_duplicates()
exporter.to_csv("output\\Exporter.csv", index=False)
MA_exporter.to_csv("output\\MA_Exporter.csv", index=False)

country_col = ['importer', 'exporter', 'contig', 'dist']
MA_country_col = ['importer', 'exporter', 'MA_contig', 'MA_dist']
country = data[country_col].drop_duplicates()
MA_country = data[MA_country_col].drop_duplicates()
country.to_csv("output\\Country.csv", index=False)
MA_country.to_csv("output\\MA_Country.csv", index=False)



col = set(MA_country + MA_importer + MA_exporter + MA_trade)
a = data.loc[data.year>=2015, list(col)]
print(f'Number of Nan values after the transformation: {a.isna().sum().sum():,} cells,  {100* a.isna().sum().sum()/(a.shape[0]*a.shape[1]):.2f} %')


# Visualizing Total Trade Value Over Time
# We are marking the Covid years but no fixed year effect for now!
df  = data.groupby('year')['value'].sum().reset_index()
df2  = data.loc[data.hsCode!=2709].groupby('year')['value'].sum().reset_index()
df.columns = ['year', 'value']
df2.columns = ['year', 'value']
fig = px.line(df, x='year', y='value', title='Total Trade Value (3-year MA) Over Time')
fig.add_scatter(x=df2['year'], y=df2['value'], mode='lines+markers', name='Total Trade Excluding oil')
fig.update_traces(mode='lines+markers')
fig.update_layout(xaxis_title='Year', yaxis_title='Total Trade Value (in 1,000 USD)')
fig.show()



