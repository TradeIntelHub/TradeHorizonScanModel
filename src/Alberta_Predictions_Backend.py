import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.model import TradeHorizonScanModel
from src.data_utils import load_maps, TradeDataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn



exporter_map, importer_map, country_map = load_maps(
        '../TradeHorizonScan/data/MA_Exporter.csv', 
        '../TradeHorizonScan/data/MA_Importer.csv',
        '../TradeHorizonScan/data/MA_Country.csv'
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
    trd_path = '../TradeHorizonScan/data/MA_Trade.csv', 
    exp_map = exporter_map,
    imp_map = importer_map,
    cty_map = country_map,
    trd_feats = trade_feats,
    inference_mode = True,
    Alberta_path = '../TradeHorizonScan/data/MA_Trade_Alberta.csv'
)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TradeHorizonScanModel(n_hs = len(dataset.hs_map),
    dim_trd = len(trade_feats),
    dim_exp = next(iter(exporter_map.values())).shape[0],
    dim_imp = next(iter(importer_map.values())).shape[0],
    dim_cty = next(iter(country_map.values())).shape[0]).to(device)

checkpoint = torch.load('../TradeHorizonScan/models/checkpoint165.pth')
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
all_train_losses = checkpoint['all_train_losses']
all_val_losses = checkpoint['all_val_losses']
_ = model.eval()





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
                index=['Country', 'Actual_2024_Trade', 'Model_Predicted_Trade', 'MA_Value', 'Importers_Total_Imports_MA']).T
    Results['Adjusted_Predicted_Trade'] = Results['Model_Predicted_Trade'] * (Results['Actual_2024_Trade'].sum() / Results['Model_Predicted_Trade'].sum())
    RoW_Trade = 0
    if HS4Code in dataset.rest_of_the_world_hsCode_to_Trade_Value_2024.keys():
        RoW_Trade = dataset.rest_of_the_world_hsCode_to_Trade_Value_2024[HS4Code]
    new_row = pd.DataFrame([{
        'Country': 'RoW',
        'Actual_2024_Trade': RoW_Trade,
        'Model_Predicted_Trade': np.nan,
        'MA_Value': np.nan,
        'Importers_Total_Imports_MA': np.nan,
        'Adjusted_Predicted_Trade':  np.nan
    }])
    Results = pd.concat([Results, new_row], ignore_index=True)
    # IF there is a country that is missing from the Results dataframe, we add it with NaN values
    for country in set(dataset.code_to_country.values()) - set(['Canada', 'Alberta', 'Rest of the World']):
        if country not in Results['Country'].values:
            new_row = pd.DataFrame([{
                'Country': country,
                'Actual_2024_Trade': np.nan,
                'Model_Predicted_Trade': np.nan,
                'MA_Value': np.nan,
                'Importers_Total_Imports_MA': np.nan,
                'Adjusted_Predicted_Trade':  np.nan
            }])
            Results = pd.concat([Results, new_row], ignore_index=True)
    Results.sort_values(by='Actual_2024_Trade', ascending=False, inplace=True)
    Results.reset_index(drop=True, inplace=True)
    Results.iloc[:, 1:] = Results.iloc[:, 1:] * 1000 # Now everything is in dollars (not in thousands of dollars anymore)
    return Results

def plot_trade_predictions(Results, name_of_commodity):
    Results = Results.set_index('Country')
    Results = Results.reindex(
        [country for country in Results.index if country != 'RoW'] + ['RoW']
    ).reset_index()
    customdata = np.stack([
    Results['Importers_Total_Imports_MA'].values,
    Results['Actual_2024_Trade'].values / Results['Importers_Total_Imports_MA'].values * 100,
    Results['Adjusted_Predicted_Trade'].values / Results['Importers_Total_Imports_MA'].values * 100
                            ], axis=-1)
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=Results['Country'],
        y=Results['Actual_2024_Trade'],
        name='Actual 2024 Trade',
        marker_color='steelblue',
        customdata=customdata,
        hovertemplate=(
            'Country: %{x}<br>' +
            'Actual Trade: $%{y:,.0f}<br>' +
            'Total Imports (MA): $%{customdata[0]:,.0f}<br>' +
            'Share of Total: %{customdata[1]:.1f}%<extra></extra>'
        )
    ))

    fig.add_trace(go.Bar(
        x=Results['Country'],
        y=Results['Adjusted_Predicted_Trade'],
        name='Adjusted Predicted Trade',
        marker_color='darkorange',
        customdata=customdata,
        hovertemplate=(
            'Country: %{x}<br>' +
            'Predicted Trade: $%{y:,.0f}<br>' +
            'Total Imports (MA): $%{customdata[0]:,.0f}<br>' +
            'Share of Total: %{customdata[2]:.1f}%<extra></extra>'
        )
    ))

    fig.update_layout(
        title={
            'text': f'Actual vs Adjusted Predicted Trade for HS Code {name_of_commodity}, (CAD)',
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
            title='Trade Value',
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





HS4Code = 1001
plot_trade_predictions(get_trade_predictions(HS4Code), str(HS4Code))





# Getting the total trade for all HS4 codes
dfs = []
for HS4Code in dataset.Alberta_df.hsCode.unique():
    dfs.append(get_trade_predictions(HS4Code))
df_total = sum(df.set_index('Country').fillna(0) for df in dfs).reset_index()
df_total.sort_values(by='Actual_2024_Trade', ascending=False, inplace=True)
df_total = df_total.round(0)
df_total.loc[df_total.Country!="RoW"].sum()
plot_trade_predictions(df_total, 'Total')



