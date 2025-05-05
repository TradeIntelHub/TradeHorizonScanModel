from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from model import TradeHorizonScanModel
from torcheval.metrics import R2Score

def cross_validate(
    dataset,
    hs_map_size: int,
    yr_map_size: int,
    dim_trade: int,
    dim_exp: int,
    dim_imp: int,
    dim_cty: int,
    k_splits: int = 9,
    batch_size: int = 1024,
    lr: float = 1e-3,
    epochs: int = 20,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> tuple[float, float,float, float]:
    kfold = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    fold_losses = []
    fold_r2s = []

    for _, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=4)

        model = TradeHorizonScanModel(
            n_hs = hs_map_size,
            n_yr = yr_map_size,
            dim_trd = dim_trade,
            dim_exp = dim_exp,
            dim_imp = dim_imp,
            dim_cty = dim_cty
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(1, epochs + 1):
            model.train()
            for h_idx, y_idx, tx, ex, im, ct, y in train_loader:
                h_idx, y_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx,y_idx,tx,ex,im,ct,y)]
                preds = model(h_idx,y_idx,tx,ex,im,ct)
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # validation
        val_loss = 0.0
        r2_metric = R2Score().to(device) 
        model.eval()
        with torch.no_grad():
            for h_idx, y_idx, tx, ex, im, ct, y in val_loader:
                h_idx, y_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx,y_idx,tx,ex,im,ct,y)]
                preds = model(h_idx, y_idx, tx, ex, im, ct)
                _ = r2_metric.update(preds, y)
                val_loss += criterion(preds, y).item() * y.size(0)
        val_loss /= len(val_idx)
        fold_losses.append(val_loss)
        fold_r2s.append(r2_metric.compute())

    mean_loss = float(np.mean(fold_losses))
    std_loss  = float(np.std(fold_losses))
    return mean_loss, std_loss
