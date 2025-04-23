import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

os.chdir("C:\\Users\\saeed.shadkam\\OneDrive - Government of Alberta\\Desktop\\ComTrade\\Other Tables\\")
data = pd.read_csv("3- Diversification_Project_Preprocessed.csv", low_memory=False)
data.sort_values(by=['importer', 'exporter', 'hsCode', 'year'], inplace=True)
data.reset_index()
a = data.loc[data.year>=2015]
print(f'Number of Nan values before the transformation: {a.isna().sum().sum()} cells,  {100* a.isna().sum().sum()/(a.shape[0]*a.shape[1])} %')


data = data.loc[data.exporter == 32]



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
col = ['AvgUnitPriceofImporterToWorld', 'TotalImportofCmdbyReporter']
df = data[['importer', 'hsCode', 'year', 'AvgUnitPriceofExporterToWorld', 'TotalExportofCmdbyPartner']].drop_duplicates()
df = df.sort_values(by=['importer', 'hsCode', 'year'])
df = df.set_index('year').groupby(['importer', 'hsCode'])[col].rolling(window=3, min_periods=1).mean().reset_index()
df.columns = ['importer', 'hsCode', 'year', 'MA_AvgUnitPriceofImporterToWorld', 'MA_TotalImportofCmdbyReporter']
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
print(f'Number of Nan values before the transformation: {a.isna().sum().sum()} cells,  {100* a.isna().sum().sum()/(a.shape[0]*a.shape[1])} %')

# Saving the output files
trade_col = ['hsCode', 'year', 'importer', 'exporter', 'value', 'AvgUnitPrice', 'AvgUnitPriceofImporterFromWorld', 'TotalImportofCmdbyReporter',
             'AvgUnitPriceofExporterToWorld', 'TotalExportofCmdbyPartner','Trade_Complementarity','Partner_Revealed_Comparative_Advantage', 'Liberalising', 'Harmful']
MA_trade_col = ['hsCode', 'year', 'importer', 'exporter','MA_value', 'MA_AvgUnitPrice', 'MA_AvgUnitPriceofImporterFromWorld', 'MA_TotalImportofCmdbyReporter',
                'MA_AvgUnitPriceofExporterToWorld', 'MA_TotalExportofCmdbyPartner','MA_Trade_Complementarity','MA_Partner_Revealed_Comparative_Advantage', 'MA_Liberalising', 'MA_Harmful']
trade = data[trade_col].drop_duplicates()
MA_trade = data[MA_trade_col].drop_duplicates()
trade.to_csv("Final Tables\\Trade.csv", index=False)
MA_trade.to_csv("Final Tables\\MA_Trade.csv", index=False)

importer_col = ['importer', 'year', 'Theil_Importer_Concentration', 'GDPPerCapita_importer', 'TariffRatesAllProductsWeigthedAverage_importer', 'GeopoliticalIndex_importer', 'ConsumerPriceIndex_importer']
MA_importer_col = ['importer', 'year', 'MA_Theil_Importer_Concentration', 'MA_GDPPerCapita_importer', 'MA_TariffRatesAllProductsWeigthedAverage_importer', 'MA_GeopoliticalIndex_importer', 'MA_ConsumerPriceIndex_importer']
importer = data[importer_col].drop_duplicates()
MA_importer = data[MA_importer_col].drop_duplicates()
importer.to_csv("Final Tables\\Importer.csv", index=False)
MA_importer.to_csv("Final Tables\\MA_Importer.csv", index=False)

exporter_col = ['exporter', 'year', 'Theil_Exporter_Concentration', 'GDPPerCapita_exporter', 'TariffRatesAllProductsWeigthedAverage_exporter', 'GeopoliticalIndex_exporter', 'ConsumerPriceIndex_exporter']
MA_exporter_col = ['exporter', 'year', 'MA_Theil_Exporter_Concentration', 'MA_GDPPerCapita_exporter', 'MA_TariffRatesAllProductsWeigthedAverage_exporter', 'MA_GeopoliticalIndex_exporter', 'MA_ConsumerPriceIndex_exporter']
exporter = data[exporter_col].drop_duplicates()
MA_exporter = data[MA_exporter_col].drop_duplicates()
exporter.to_csv("Final Tables\\Exporter.csv", index=False)
MA_exporter.to_csv("Final Tables\\MA_Exporter.csv", index=False)

country_col = ['importer', 'exporter', 'contig', 'dist']
MA_country_col = ['importer', 'exporter', 'MA_contig', 'MA_dist']
country = data[country_col].drop_duplicates()
MA_country = data[MA_country_col].drop_duplicates()
country.to_csv("Final Tables\\Country.csv", index=False)
MA_country.to_csv("Final Tables\\MA_Country.csv", index=False)



