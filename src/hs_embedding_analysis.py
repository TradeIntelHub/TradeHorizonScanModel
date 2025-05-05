from src.model import TradeHorizonScanModel
from src.data_utils import load_maps, TradeDataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np

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
model = TradeHorizonScanModel(n_hs = len(dataset.hs_map),
    dim_trd = len(trade_feats),
    dim_exp = next(iter(exporter_map.values())).shape[0],
    dim_imp = next(iter(importer_map.values())).shape[0],
    dim_cty = next(iter(country_map.values())).shape[0])

checkpoint = torch.load('../TradeHorizonScan/models/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
train_losses = checkpoint['loss']


model.eval()
keys = list(dataset.hs_map.keys())
keys.sort()
values = [dataset.hs_map[key] for key in keys]
with torch.no_grad():
    hs_emb = model.hs_emb(torch.tensor(values)).cpu().numpy()
hs_embedding = dict(zip(keys, hs_emb))


hs_embedding['9702']
hs_embedding = {str(k).zfill(4): v for k, v in hs_embedding.items()}

from collections import defaultdict
prefix_groups = defaultdict(list)
for code in hs_embedding.keys():
    prefix = code[:2]
    prefix_groups[prefix].append(code)

prefix_groups


from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
keys = sorted(hs_embedding.keys())
embeddings = np.stack([hs_embedding[k] for k in keys])
cos_sim_matrix = cosine_similarity(embeddings)
key_to_idx = {k: i for i, k in enumerate(keys)}

within_group_sims = []
out_group_sims = []
for prefix, codes in prefix_groups.items():
    indices = [key_to_idx[c] for c in codes if c in key_to_idx]
    for i in indices:
        for j in range(len(keys)):
            if i == j:
                continue  # skip self-comparison
            sim = cos_sim_matrix[i, j]
            
            if keys[j][:2] == prefix:
                within_group_sims.append(sim)
            else:
                out_group_sims.append(sim)

avg_within = np.mean(abs(np.array(within_group_sims)))
avg_out = np.mean(abs(np.array(out_group_sims)))

print(f"Avg cosine similarity within groups: {avg_within:.4f}")
print(f"Avg cosine similarity out of groups: {avg_out:.4f}")