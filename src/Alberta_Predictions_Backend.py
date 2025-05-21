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



dataset.Alberta_df.loc[(dataset.Alberta_df.hsCode == 2709) & (dataset.Alberta_df.year == 2023)]

idx = dataset.Alberta_df.loc[(dataset.Alberta_df.hsCode == 2709) & (dataset.Alberta_df.year == 2023) & (dataset.Alberta_df.importer == 840)].index[0]

with torch.no_grad():
    (h_idx, tx, ex, im, ct, y) = dataset.__getitem__(idx)
    h_idx, tx, ex, im, ct, y = [t.unsqueeze(0).to(device) for t in (h_idx, tx,ex,im,ct,y)]
    y_pred = model(h_idx, tx, ex, im, ct)
    y_pred = y_pred.cpu()

y_pred = 6586867
actual = dataset.Alberta_df.loc[idx, 'MA_value']
(actual - y_pred)/actual