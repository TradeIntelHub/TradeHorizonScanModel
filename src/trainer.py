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
    #yr_map_size: int,
    dim_trade: int,
    dim_exp: int,
    dim_imp: int,
    dim_cty: int,
    k_splits: int = 9,
    batch_size: int = 1024,
    lr: float = 1e-3,
    epochs: int = 20,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[float, float]:
    kfold = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    fold_losses = []
    fold_r2s = []

    for _, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=4)

        model = TradeHorizonScanModel(
            n_hs = hs_map_size,
            #n_yr = yr_map_size,
            dim_trd = dim_trade,
            dim_exp = dim_exp,
            dim_imp = dim_imp,
            dim_cty = dim_cty
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(1, epochs + 1):
            model.train()
            for h_idx, tx, ex, im, ct, y in train_loader: #remove y_idx to match with the model
                h_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx,tx,ex,im,ct,y)]#remove y_idx to match with the model
                preds = model(h_idx,tx,ex,im,ct)
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # validation
        val_loss = 0.0
        r2_metric = R2Score().to(device) 
        model.eval()
        with torch.no_grad():
            for h_idx, tx, ex, im, ct, y in val_loader:#remove y_idx to match with the model
                h_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx, tx,ex,im,ct,y)]#remove y_idx to match with the model
                preds = model(h_idx, tx, ex, im, ct)
                _ = r2_metric.update(preds, y)
                val_loss += criterion(preds, y).item() * y.size(0)
        val_loss /= len(val_idx)
        fold_losses.append(val_loss)
        fold_r2s.append(r2_metric.compute())

    #print MSE
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    folds = list(range(1, k_splits + 1))
    plt.plot(folds, fold_losses, marker='o', linestyle='-', color='b', label='Validation Loss')
    plt.title('Validation Loss (MSE) Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    plt.legend()
    
    # Print R² Score
    plt.subplot(1, 2, 2)
    plt.plot(folds, fold_r2s, marker='o', linestyle='-', color='g', label='R² Score')
    plt.title('R² Score Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()

    mean_loss = float(np.mean(fold_losses))
    std_loss  = float(np.std(fold_losses))
    mean_r2 = float(np.mean(fold_r2s))
    std_r2 = float(np.std(fold_r2s))
    return mean_loss, std_loss
