import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from src.model import TradeHorizonScanModel


#Retreiving the Alberta Trade Data
import pyodbc
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=c-goa-sql-10593;'
    'DATABASE=JET_TRDAT_UAT;'
    'Trusted_Connection=yes;'
)
data = []
for year in range(2013, 2025):
    exports = pd.read_sql(f"SELECT * FROM dbo.statCanExportDataMonthly{year}", conn)
    exports['hs4Code'] = exports['hs6Code'].astype(str).str.zfill(6).str[:4]
    exports = exports.groupby(['hs4Code', 'ctyCode', 'provCode'])[['Value', 'Quantity']].sum().reset_index()
    exports['Year'] = year
    exports = exports.loc[: ,[ 'Year', 'hs4Code', 'ctyCode', 'provCode', 'Value', 'Quantity']]
    exports.rename(columns={'ctyCode': 'Country', 'provCode': 'Province'}, inplace=True)
    data.append(exports)
exports = pd.concat(data, ignore_index=True)



Alberta = exports[exports['Province'] == 'AB'].reset_index(drop=True)
Alberta.drop(columns=['Province'], inplace=True)
Alberta['AvgUnitPrice'] = Alberta['Value'] / Alberta['Quantity']
Alberta['AvgUnitPriceFlags'] = Alberta.AvgUnitPrice == np.inf
Alberta['AvgUnitPrice'] = Alberta['AvgUnitPrice'].replace(np.inf, 0)

Canada = exports.groupby(['Year', 'hs4Code', 'Country'])[['Value', 'Quantity']].sum().reset_index()
Canada.rename(columns={'Value': 'CA_Value', 'Quantity': 'CA_Quantity'}, inplace=True)
Canada['CA_AvgUnitPrice'] = Canada['CA_Value'] / Canada['CA_Quantity']
Canada['CA_AvgUnitPriceFlags'] = Canada.CA_AvgUnitPrice == np.inf
Canada['CA_AvgUnitPrice'] = Canada['CA_AvgUnitPrice'].replace(np.inf, 0)

# Merge with Ablerta data
Alberta = Alberta.merge(Canada, on=['Year', 'hs4Code', 'Country'], how='left')
Alberta

pd.read_csv('../TradeHorizonScan/src/Pre-processing/1- CEPII_Processed_HS4_2013_2023.csv').head()
import os
print(os.getcwd())

























# Loading the datasets


# Loading the model

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TradeHorizonScanModel(n_hs = len(dataset.hs_map),
    dim_trd = len(trade_feats),
    dim_exp = next(iter(exporter_map.values())).shape[0],
    dim_imp = next(iter(importer_map.values())).shape[0],
    dim_cty = next(iter(country_map.values())).shape[0]).to(device)


checkpoint = torch.load('../TradeHorizonScan/models/checkpoint165.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
all_train_losses = checkpoint['all_train_losses']
all_val_losses = checkpoint['all_val_losses']
_ = model.eval()


