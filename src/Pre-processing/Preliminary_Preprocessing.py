import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
# This code is designed for the first version of the Xinyue files which had all the variables
# I comment out the lines related to those variables and run it for the Selected Variables version
# Load the data


os.chdir(os.path.join(os.getcwd(), 'src', 'Pre-processing', 'data'))
df = pd.read_csv("2- Diversification_Project_Raw.csv", low_memory=False)

# Making sure data types are reasonable
for col in df.columns:
    print(col,":", type(df.loc[:, col].iloc[0]), ":", df.loc[:, col].iloc[0])

# Rename some columns
#df.rename(columns = {' GDPPerCapita _exporter':'GDPPerCapita_exporter'}, inplace = True)
#df.rename(columns = {' GDPPerCapita _importer':'GDPPerCapita_importer'}, inplace = True)

# Preprocessing the data
# 1. Remove the dollar sign
df['GDPPerCapita_exporter'] = df['GDPPerCapita_exporter'].str.replace('$', '').str.replace(',', '').str.replace(' ', '')
df['GDPPerCapita_importer'] = df['GDPPerCapita_importer'].str.replace('$', '').str.replace(',', '').str.replace(' ', '')
#df['ConsumerPriceIndex_importer'] = df['ConsumerPriceIndex_importer'].str.replace('$', '').str.replace(" N/A", "").str.replace(' ', '')
#df['ConsumerPriceIndex_exporter'] = df['ConsumerPriceIndex_exporter'].str.replace('$', '').str.replace(" N/A", "").str.replace(' ', '')

# 2. Convert the data to numeric
df['GDPPerCapita_exporter'] = pd.to_numeric(df['GDPPerCapita_exporter'])
df['GDPPerCapita_exporter'] = df['GDPPerCapita_exporter'].astype(int)
df['GDPPerCapita_importer'] = pd.to_numeric(df['GDPPerCapita_importer'])
#df['ConsumerPriceIndex_importer'] = pd.to_numeric(df['ConsumerPriceIndex_importer'])
#df['ConsumerPriceIndex_exporter'] = pd.to_numeric(df['ConsumerPriceIndex_exporter'])

# Columns to drop
'''
col = ['year', 'importer', 'exporter', 'hsCode', 'GDP_importer', 'Population_importer', 'GDP_exporter', 'Population_exporter']
col2 = ['LogisticsPerformanceIndex(LPI)_importer', 'LogisticsPerformanceIndex(LPI)_exporter', 'RuleOfLaw_importer', 'RuleOfLaw_exporter', 
       'PoliticalStability_importer', 'PoliticalStability_exporter', 'SchoolEnrollment_importer', 'SchoolEnrollment_exporter']
col3 = col + col2'
'''
col3 = ['year', 'importer', 'exporter', 'hsCode']
corr_matrix = df.drop(col3, axis=1).corr()

# Heatmap of the correlation matrix using plotly
fig = px.imshow(corr_matrix)
fig.show()

# Finding the average and Max CORR for each variable
avg_corr = (abs(corr_matrix).sum() - 1)/len(corr_matrix)
max_corr = abs(corr_matrix).replace(1, 0).max()
avg_corr = avg_corr.sort_values(ascending=False)
max_corr = max_corr.sort_values(ascending=False)
fig = go.Figure(data=[
    go.Bar(name='Average Correlation', x=max_corr.index, y=max_corr)
])
fig.show()


# VIF analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X = df.drop(col3, axis=1)
X = add_constant(X)
X = X.dropna()
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data = vif_data[vif_data['feature'] != 'const']
vif_data = vif_data.sort_values(by='VIF', ascending=False)
fig = go.Figure(data=[
    go.Bar(name='VIF', x=vif_data['feature'], y=vif_data['VIF'])
])
fig.update_layout(title='VIF Values', xaxis_title='Features', yaxis_title='VIF')
fig.show()

# Final Step:
# Decided columns to go to the next stage of the analysis
to_be_removed = set(col3) - set(['year', 'importer', 'exporter', 'hsCode'])
final= df.drop(to_be_removed, axis=1)


# I will replace nan values in UnitPrice columnS with 0s and add a new columns called UnitPriceFlags to indicate the rows that were replaced
'''
final['AvgUnitPriceFlags'] = np.where(final['AvgUnitPrice'].isna(), True, False)
final['AvgUnitPrice'] = final['AvgUnitPrice'].fillna(0)

final['AvgUnitPriceofImporterFromWorldFlags'] = np.where(final['AvgUnitPriceofImporterFromWorld'].isna(), True, False)
final['AvgUnitPriceofImporterFromWorld'] = final['AvgUnitPriceofImporterFromWorld'].fillna(0)

final['AvgUnitPriceofExporterToWorldFlags'] = np.where(final['AvgUnitPriceofExporterToWorld'].isna(), True, False)
final['AvgUnitPriceofExporterToWorld'] = final['AvgUnitPriceofExporterToWorld'].fillna(0)
'''
# Taking care of null values in Harmful and Liberalising columns which means they are zeros
final['Harmful'] = final['Harmful'].fillna(0)
final['Liberalising'] = final['Liberalising'].fillna(0)

final.to_csv("3- Diversification_Project_Preprocessed.csv", index=False)