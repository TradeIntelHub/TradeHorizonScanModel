import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.model import TradeHorizonScanModel
from src.data_utils import load_maps, TradeDataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import pyodbc
conn = pyodbc.connect(
                        'DRIVER={ODBC Driver 17 for SQL Server};'
                        'SERVER=c-goa-sql-10593;'
                        'DATABASE=JET_TRDAT_UAT;'
                        'Trusted_Connection=yes;'
                    )

Country_codes = pd.read_sql("SELECT * FROM dbo.countryCodesDescriptionStatCanUNComTradeUSCensusBureau", conn)


exporter_map, importer_map, country_map = load_maps(
        '../TradeHorizonScanModel/data/MA_Exporter.csv', 
        '../TradeHorizonScanModel/data/MA_Importer.csv',
        '../TradeHorizonScanModel/data/MA_Country.csv'
    )
trade_feats: List[str] = [
        'MA_AvgUnitPrice',
        'MA_AvgUnitPriceFlags',
        'MA_AvgUnitPriceofImporterFromWorld',
        'MA_AvgUnitPriceofImporterFromWorldFlags',
        'MA_TotalImportofCmdbyReporter',
        'MA_AvgUnitPriceofExporterToWorld',
        'MA_AvgUnitPriceofExporterToWorldFlags',
        'MA_TotalExportofCmdbyPartner',
        'MA_Trade_Complementarity',
        'MA_Partner_Revealed_Comparative_Advantage',
        'MA_Liberalising',
        'MA_Harmful',
        'Covid'
    ]
dataset = TradeDataset(
    trd_path = '../TradeHorizonScanModel/data/MA_Trade.csv', 
    exp_map = exporter_map,
    imp_map = importer_map,
    cty_map = country_map,
    trd_feats = trade_feats,
    inference_mode = True,
    Alberta_path = '../TradeHorizonScanModel/data/MA_Trade_Alberta.csv',
    sql_conn = conn
)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TradeHorizonScanModel(n_hs = len(dataset.hs_map),
    dim_trd = len(trade_feats),
    dim_exp = next(iter(exporter_map.values())).shape[0],
    dim_imp = next(iter(importer_map.values())).shape[0],
    dim_cty = next(iter(country_map.values())).shape[0]).to(device)

checkpoint = torch.load('../TradeHorizonScanModel/models/checkpoint243.pth')
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
all_train_losses = checkpoint['all_train_losses']
all_val_losses = checkpoint['all_val_losses']
_ = model.eval()


exchange_rate = pd.read_sql(f"SELECT * FROM statCanMonthlyCurrencyExchangeRatesUSDtoCAD", conn)
exchange_rate = exchange_rate.sort_values('YearMonth').reset_index(drop=True)
USDCAD = exchange_rate.iloc[-1, :]['ExchangeRateUSDtoCAD']


def get_trade_predictions(HS4Code):
    idx = list(dataset.Alberta_df.loc[(dataset.Alberta_df.hsCode == HS4Code)].index)
    predicted_trades = []
    actual_2024_trades = []
    actual_MA_values = []
    countries = []
    total_import_MA = []
    Results = pd.DataFrame()
    with torch.no_grad():
        for i in idx:
            (h_idx, tx, ex, im, ct, y) = dataset.__getitem__(i)
            h_idx, tx, ex, im, ct = [t.unsqueeze(0).to(device) for t in (h_idx, tx, ex, im, ct)]
            y_pred = model(h_idx, tx, ex, im, ct)
            predicted_trades.append(y_pred.item())
            actual_MA_values.append(y.item())
            actual_2024_trades.append(dataset.Alberta_df.iloc[i].Actual_Alberta_2024_Values)
            countries.append(dataset.code_to_country[dataset.Alberta_df.iloc[i].importer])
            total_import_MA.append(dataset.Alberta_df.iloc[i].MA_TotalImportofCmdbyReporter)
    Results = pd.DataFrame((countries,actual_2024_trades, predicted_trades, actual_MA_values, total_import_MA ), 
                index=['Country', 'Actual_2024_Trade_CAD', 'Model_Predicted_Trade_USD', 'MA_Value_USD', 'Importers_Total_Imports_MA_USD']).T
    Results['Model_Predicted_Trade_CAD'] = Results['Model_Predicted_Trade_USD'] * USDCAD
    Results['Adjusted_Predicted_Trade_CAD'] = np.nan

    RoW_Trade = 0
    if HS4Code in dataset.rest_of_the_world_hsCode_to_Trade_Value_2024.keys():
        RoW_Trade = dataset.rest_of_the_world_hsCode_to_Trade_Value_2024[HS4Code]
    new_row = pd.DataFrame([{
        'Country': 'RoW',
        'Actual_2024_Trade_CAD': RoW_Trade,
        'Model_Predicted_Trade_USD': np.nan,
        'MA_Value_USD': np.nan,
        'Importers_Total_Imports_MA_USD': np.nan,
        'Adjusted_Predicted_Trade_CAD':  np.nan
    }])
    Results = pd.concat([Results, new_row], ignore_index=True)
    # IF there is a country that is missing from the Results dataframe, we add it with NaN values
    for country in set(dataset.code_to_country.values()) - set(['Canada', 'Alberta', 'Rest of the World']):
        if country not in Results['Country'].values:
            new_row = pd.DataFrame([{
                'Country': country,
                'Actual_2024_Trade_CAD': np.nan,
                'Model_Predicted_Trade_USD': np.nan,
                'MA_Value_USD': np.nan,
                'Importers_Total_Imports_MA_USD': np.nan,
                'Adjusted_Predicted_Trade_CAD':  np.nan
            }])
            Results = pd.concat([Results, new_row], ignore_index=True)
    Results['Adjusted_Predicted_Trade_CAD'] = Results['Model_Predicted_Trade_USD'] * (Results['Actual_2024_Trade_CAD'].sum() / Results['Model_Predicted_Trade_USD'].sum())
    Results.sort_values(by='Actual_2024_Trade_CAD', ascending=False, inplace=True)
    Results.reset_index(drop=True, inplace=True)
    Results.iloc[:, 1:] = Results.iloc[:, 1:] * 1000 # Now everything is in dollars (not in thousands of dollars anymore)
    assert math.isclose(Results['Adjusted_Predicted_Trade_CAD'].sum(), Results['Actual_2024_Trade_CAD'].sum(), abs_tol =5), "The adjusted predicted trade does not match the actual trade"
    return Results

def plot_trade_predictions(Results, name_of_commodity, HS4Code, include_RoW=True, include_raw_model_predictions=False):
    
    Results = Results.set_index('Country')
    Results = Results.reindex(
        [country for country in Results.index if country != 'RoW'] + ['RoW']
                                ).reset_index()

    if not include_RoW:
        Results = Results.loc[Results.Country != 'RoW',:]
    
    customdata = np.stack([
    Results['Importers_Total_Imports_MA_USD'].values * USDCAD,
    Results['Actual_2024_Trade_CAD'].values /(Results['Importers_Total_Imports_MA_USD'].values * USDCAD) * 100,
    Results['Adjusted_Predicted_Trade_CAD'].values /(Results['Importers_Total_Imports_MA_USD'].values * USDCAD) * 100,
    Results['Model_Predicted_Trade_CAD'].values /(Results['Importers_Total_Imports_MA_USD'].values * USDCAD) * 100,
    Results['Actual_2024_Trade_CAD'].sum() * np.ones(Results.shape[0])  # Alberta's total exports
                            ], axis=-1)
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=Results['Country'],
        y=Results['Actual_2024_Trade_CAD'],
        name='Actual 2024 Trade',
        marker_color='steelblue',
        customdata=customdata,
        hovertemplate=(
            'Country: %{x}<br>' +
            'Actual Trade: $%{y:,.0f}<br>' +
            'Total Imports (3yr-MA): $%{customdata[0]:,.0f}<br>' +
            'Share of Total: %{customdata[1]:.1f}%<br>' + 
            "Alberta's Total Exports: $%{customdata[4]:,.0f}<extra></extra>" 
        )
    ))

    fig.add_trace(go.Bar(
        x=Results['Country'],
        y=Results['Adjusted_Predicted_Trade_CAD'],
        name='Model Predicted Trade',
        marker_color='darkorange',
        customdata=customdata,
        hovertemplate=(
            'Country: %{x}<br>' +
            'Predicted Trade: $%{y:,.0f}<br>' +
            'Total Imports (3yr-MA): $%{customdata[0]:,.0f}<br>' +
            'Share of Total: %{customdata[2]:.1f}%<br>'+ 
            "Alberta's Total Exports: $%{customdata[4]:,.0f}<extra></extra>" 
        )
    ))

    if include_raw_model_predictions:
        fig.add_trace(go.Bar(
            x=Results['Country'],
            y=Results['Model_Predicted_Trade_CAD'],
            name='Raw Model Predicted Trade',
            marker_color='lightgreen',
            customdata=customdata,
            hovertemplate=(
                'Country: %{x}<br>' +
                'Raw Model Predicted Trade: $%{y:,.0f}<br>' +
                'Total Imports (3yr-MA): $%{customdata[0]:,.0f}<br>' +
                'Share of Total: %{customdata[3]:.1f}%<br>>'+ 
            "Alberta's Total Exports: $%{customdata[4]:,.0f}<extra></extra>" 
            )
        ))


    fig.update_layout(
        title={
            'text': f'Actual vs Model Predicted Trade for {name_of_commodity} (HS4: {HS4Code}, CAD)',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=22)
        },
        xaxis=dict(
            title='Country',
            tickangle=-45,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title='Export Value',
            tickfont=dict(size=12)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12)
        ),
        barmode='group',  # ← switch from 'stack' to 'group'
        template='plotly_white',
        margin=dict(l=80, r=50, t=100, b=100),
        height=600,
        width=1000
    )
    fig.show()

##################################
# Ensure Alberta_df has high quality data
# Investigate the model predictions before adjusting them for the Alberta total supply value. See if they make sense before adjusting.
# Train a model on the Top 50 countries instead of top alberta partners!
##################################






# Add HS4 codes to the title
HS4Code = (201, "Fresh or Chilled Beef")
HS4Code = (1001,  "Wheat")
HS4Code = (2709, "Crude Oil")
HS4Code = (1205, "Canola")
HS4Code = (9603, "BROOMS")
HS4Code = (409, "Honey")
HS4Code = (1902, "Pasta")
HS4Code = (1004, "Oats")
HS4Code = (102, "Cattle")
HS4Code = (3901, "Polymers of Ethylene")


results = get_trade_predictions(HS4Code[0])
plot_trade_predictions(results, HS4Code[1], HS4Code[0], include_RoW=False, include_raw_model_predictions=False)



# Getting the total trade for all HS4 codes
dfs = []
for HS4Code in dataset.Alberta_df.hsCode.unique():
    try:
        a = get_trade_predictions(HS4Code)
    except Exception as e:
        print(f"HS4Code {HS4Code}: is not part of the UnComtrade dataset, skipping the model prediction for this HS4Code")
        print('Will put the Actual numbers as the model prediction to keep the total trade value consistent')
        a = dataset.Alberta_df[dataset.Alberta_df.hsCode == HS4Code]
        a = a[['importer', 'Actual_Alberta_2024_Values']]
        a = a.replace({'importer': dataset.code_to_country})
        a.rename(columns={'importer': 'Country', 'Actual_Alberta_2024_Values': 'Actual_2024_Trade_CAD'}, inplace=True)
        a['Model_Predicted_Trade_CAD'] = a['Actual_2024_Trade_CAD'] 
        a['Adjusted_Predicted_Trade_CAD'] = a['Actual_2024_Trade_CAD'] 
        a['Importers_Total_Imports_MA_USD'] = a['Actual_2024_Trade_CAD'] 
        a['Model_Predicted_Trade_USD'] = a['Actual_2024_Trade_CAD'] 
        a['MA_Value_USD'] = a['Actual_2024_Trade_CAD'] 
        a = a[['Country', 'Actual_2024_Trade_CAD', 'Model_Predicted_Trade_USD', 'MA_Value_USD', 'Importers_Total_Imports_MA_USD','Model_Predicted_Trade_CAD', 'Adjusted_Predicted_Trade_CAD']]
        a.reset_index(drop=True, inplace=True)

    a['HS4Code'] = HS4Code
    dfs.append(a)

df_total = pd.concat(dfs, ignore_index=True)
df_total = df_total.groupby('Country', as_index=False).sum()
df_total.drop(columns=['HS4Code'], inplace=True)
df_total.sort_values(by='Actual_2024_Trade_CAD', ascending=False, inplace=True)
df_total.sum()
plot_trade_predictions(df_total, 'Total', "Total", include_RoW=False, include_raw_model_predictions=False)

# Let's take a look at the aggregated difference betweeen "Model Predicted Trade CAD" and the "Adjusted Predicted Trade CAD"
plot_trade_predictions(df_total, 'Total', "Total", include_RoW=False, include_raw_model_predictions=True)




# Researching Malaysia:
# As an example for the Presentation
dfs
df = pd.concat(dfs, ignore_index=True)
df = df.loc[df.Country == "Malaysia"]
df['diff'] = df['Adjusted_Predicted_Trade_CAD'] - df['Actual_2024_Trade_CAD']
df.sort_values(by='diff', ascending=False, inplace=True)

df.head(20)







# Researching the Polymers of Ethylene
HS4Code = (3901, "Polymers of Ethylene")
results = get_trade_predictions(HS4Code[0])
results['diff'] = results['Adjusted_Predicted_Trade_CAD'] - results['Actual_2024_Trade_CAD']
results.sort_values(by='diff', ascending=False, inplace=True)
results.head(21)

results.to_csv('../TradeHorizonScanModel/data/Polymers_of_Ethylene_Trade_Predictions.csv', index=False)