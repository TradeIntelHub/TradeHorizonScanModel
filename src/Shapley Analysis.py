from src.model import TradeHorizonScanModel
from src.data_utils import load_maps, Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import torch
from src.data_utils import load_maps, TradeDataset
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
import torch

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
    trd_feats = trade_feats
)

dataset.df = dataset.df.sample(frac=0.01, random_state=42).reset_index(drop=True)


batch_size = 2048
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TradeHorizonScanModel(n_hs = len(dataset.hs_map),
    dim_trd = len(trade_feats),
    dim_exp = next(iter(exporter_map.values())).shape[0],
    dim_imp = next(iter(importer_map.values())).shape[0],
    dim_cty = next(iter(country_map.values())).shape[0]).to(device)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)


# Loading the checkpoint if needed
checkpoint = torch.load('../TradeHorizonScan/models/checkpoint200.pth')
model.load_state_dict(checkpoint['model_state_dict'])
train_losses = checkpoint['loss']
model.eval()


import shap
explainer = shap.Explainer(model, train_loader)
shap_values = explainer(train_loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Sample a batch from train_loader to use as background
batch = next(iter(train_loader))
h_idx, tx, ex, im, ct, y = batch 
h_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx, tx, ex, im, ct, y)]




class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        h_idx, tx, ex, im, ct = x
        return self.model(h_idx, tx, ex, im, ct).squeeze()


import shap

# Background and test samples (keep small for KernelExplainer)
background = [t[:100].to('cpu') for t in (h_idx, tx, ex, im, ct)]
inputs_to_explain = [t[100:120].to('cpu') for t in (h_idx, tx, ex, im, ct)]

model(h_idx, tx, ex, im, ct).shape[1]

wrapped_model = WrappedModel(model).to('cpu')
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(inputs_to_explain)



#  early stopping
# 80 20 
# scatter plot for the whole thing
