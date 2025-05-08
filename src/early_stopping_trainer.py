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
from src.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
import time

# Loading the data and maps
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

dataset.df = dataset.df.sample(frac=0.1, random_state=7).reset_index(drop=True)

# Loading the model
lr = 1e-3
batch_size = 2048
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TradeHorizonScanModel(n_hs = len(dataset.hs_map),
    dim_trd = len(trade_feats),
    dim_exp = next(iter(exporter_map.values())).shape[0],
    dim_imp = next(iter(importer_map.values())).shape[0],
    dim_cty = next(iter(country_map.values())).shape[0]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



# train and val data loader
train_idx, val_idx = train_test_split(np.arange(len(dataset.df)), val_size=0.2, random_state=7, shuffle=True)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=8)



# Training the model with early stopping
early_stopping = EarlyStopping(patience=50, delta=0)
epochs = 20
all_train_losses = []
all_val_losses = []
start_time = time.time()
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i, (h_idx, tx, ex, im, ct, y) in enumerate(train_loader):
        h_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx,tx,ex,im,ct,y)]
        optimizer.zero_grad()
        preds = model(h_idx,tx,ex,im,ct)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * y.size(0)

    train_loss /= len(train_loader.dataset)
    all_train_losses.append(train_loss)
    # validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (h_idx, tx, ex, im, ct, y) in enumerate(val_loader):  
            h_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx, tx,ex,im,ct,y)]
            preds = model(h_idx,tx,ex,im,ct)
            loss = criterion(preds, y)
            val_loss += loss.item() * y.size(0)
    val_loss /= len(val_loader.dataset) 
    all_val_losses.append(val_loss)

    torch.save({'epoch': len(all_train_losses),
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'all_train_losses': all_train_losses,
    'all_val_losses': all_val_losses,
            }, f'../TradeHorizonScan/models/checkpoint{len(all_train_losses)}.pth')

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    end_time = time.time()
    print(f'Training time so far (epoch: {epoch+1}): {(end_time - start_time)/3600:.2f} hours')
    print(f'Estimated time remaining: {((end_time - start_time)/3600) /(epoch+1) * epochs:.2f} hours')
    print('=='* 50)

    early_stopping = EarlyStopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        torch.save({'epoch': len(all_train_losses),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'all_train_losses': all_train_losses,
            'all_val_losses': all_val_losses,
                    }, f'../TradeHorizonScan/models/best_model.pth')
        break

early_stopping.load_best_model(model)


# Add timer
# Save models
# Make sure it works
# Priting the plots