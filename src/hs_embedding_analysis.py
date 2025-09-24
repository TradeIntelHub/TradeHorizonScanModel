from src.model import TradeHorizonScanModel
from src.data_utils import load_maps, TradeDataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import json


with open('../TradeHorizonScanModel/src/model_parameters.json', 'r') as f:
    model_parameters = json.load(f)

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
    trd_feats = trade_feats
)
model = TradeHorizonScanModel(n_hs = len(dataset.hs_map),
    dim_trd = len(trade_feats),
    dim_exp = next(iter(exporter_map.values())).shape[0],
    dim_imp = next(iter(importer_map.values())).shape[0],
    dim_cty = next(iter(country_map.values())).shape[0])



checkpoint_address = '../TradeHorizonScanModel/models/' + f'{model_parameters["Best_checkpoint"]}'
checkpoint = torch.load(checkpoint_address)
model.load_state_dict(checkpoint['model_state_dict'])
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
train_losses = checkpoint['all_train_losses']
val_losses = checkpoint['all_val_losses']

model.eval()
keys = list(dataset.hs_map.keys())
keys.sort()
values = [dataset.hs_map[key] for key in keys]
with torch.no_grad():
    hs_emb = model.hs_emb(torch.tensor(values)).cpu().numpy()
hs_embedding = dict(zip(keys, hs_emb))


hs_embedding = {str(k).zfill(4): v for k, v in hs_embedding.items()}




# Applying PCA to reduce dimensions for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
hs_emb_2d = pca.fit_transform(np.stack(list(hs_embedding.values())))
hs_emb_2d = {k: v for k, v in zip(hs_embedding.keys(), hs_emb_2d)}

# Visualizing the embeddings 
import plotly.express as px
df = pd.DataFrame({
    'HSCode': list(hs_emb_2d.keys()),
    'Component 1': [v[0] for v in hs_emb_2d.values()],
    'Component 2': [v[1] for v in hs_emb_2d.values()]
})

fig = px.scatter(df, x='Component 1', y='Component 2', text='HSCode')
fig.update_traces(textposition='top center')
fig.show()












from collections import defaultdict
prefix_groups = defaultdict(list)
for code in hs_embedding.keys():
    prefix = code[:2]
    prefix_groups[prefix].append(code)



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



def get_top_n_similar_codes(hs_embedding, code, n=5):    
    target_embedding = hs_embedding[code]
    similarities = []
    
    for other_code, other_embedding in hs_embedding.items():
        if other_code != code:
            sim = cosine_similarity([target_embedding], [other_embedding])[0][0]
            similarities.append((other_code, sim))
    
    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:n]  # Return top n similar codes


get_top_n_similar_codes(hs_embedding, '0201', n=5)