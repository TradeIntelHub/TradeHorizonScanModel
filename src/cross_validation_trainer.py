<<<<<<< HEAD:src/trainer.py
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
    k_splits: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    epochs: int = 200,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[float, float, float, float]:
 #-> Tuple[float, float, float, float, List[np.ndarray]]:
    kfold = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    fold_losses = []
    fold_r2s = []
    all_fold_y = []  # store all y values for each fold
    all_fold_preds = [] # store all predictions for each fold

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1): #_
        print(f"K-Fold {fold} starts")
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
            print(f"Epoch {_} starts")
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
            #fold_y = []
            #fold_preds = [] # store y and preds for each fold in temporary list
            for h_idx, tx, ex, im, ct, y in val_loader:#remove y_idx to match with the model
                h_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx, tx,ex,im,ct,y)]#remove y_idx to match with the model
                preds = model(h_idx, tx, ex, im, ct)
                _ = r2_metric.update(preds, y)
                val_loss += criterion(preds, y).item() * y.size(0)
                #fold_y.append(y.cpu().numpy())  # colect y values for each fold
                #fold_preds.append(preds.cpu().numpy()) #collect preds for each fold
        val_loss /= len(val_idx)
        fold_losses.append(val_loss)
        fold_r2s.append(r2_metric.compute())
        #fold_y = np.concatenate(fold_y, axis=0)  # 
        #fold_preds = np.concatenate(fold_preds, axis=0)  # collect all y and preds for each fold
        #all_fold_y.append(fold_y)  # store y values in all_fold_y  for each fold
        #all_fold_preds.append(fold_preds) # store preds in all_fold_preds for each fold
        print(f"K-Fold {fold} result: (val_loss: {val_loss:.4f}, r2: {r2_metric.compute().item():.4f})")# print the result of each fold
    
    
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
    
    #plot all y and preds for each fold
    plt.figure(figsize=(6, 5))
    for i in range(k_splits):
        plt.scatter(all_fold_y[i], all_fold_preds[i], alpha=0.5, s=10, label=f'Fold {i+1}')
    plt.plot([min(np.concatenate(all_fold_y)), max(np.concatenate(all_fold_y))], 
             [min(np.concatenate(all_fold_y)), max(np.concatenate(all_fold_y))], 'r--', lw=2, label='Ideal')
    plt.title('True vs Predicted Values Across Folds')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


    #ape calculation by all y and preds for each fold
    '''
    fold_apes = []
    fold_mapes = []
    for i in range(k_splits):
        actual = all_fold_y[i]
        predicted = all_fold_preds[i]
        ape = np.abs((actual - predicted) / (actual + 1e-10)) * 100  # 加 1e-10 避免除零
        mape = np.mean(ape)
        fold_apes.append(ape)  # 存储每个折的 APE 数组
        fold_mapes.append(mape)
        print(f"Fold {i+1} MAPE: {mape:.2f}%")

    mean_mape = np.mean(fold_mapes)
    print(f"Average MAPE across folds: {mean_mape:.2f}%")
    '''

    mean_loss = float(np.mean(fold_losses))
    std_loss  = float(np.std(fold_losses))
    mean_r2 = float(np.mean(fold_r2s))
    std_r2 = float(np.std(fold_r2s))
    return mean_loss, std_loss, mean_r2, std_r2
    #return mean_loss, std_loss, mean_r2, std_r2, fold_apes
=======
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
    k_splits: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-3,
    epochs: int = 200,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[float, float, float, float]:
    kfold = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    fold_losses = []
    fold_r2s = []
    all_fold_y = []  # store all y values for each fold
    all_fold_preds = [] # store all predictions for each fold

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1): #_
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
        print(f"K-Fold {fold} result: (val_loss: {val_loss:.4f}, r2: {r2_metric.compute().item():.4f})")#_
    
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
    
    #plot all y and preds for each fold
    plt.subplot(1, 3, 3)
    for i in range(k_splits):
        plt.scatter(all_fold_y[i], all_fold_preds[i], alpha=0.5, s=10, label=f'Fold {i+1}')
    plt.plot([min(np.concatenate(all_fold_y)), max(np.concatenate(all_fold_y))], 
             [min(np.concatenate(all_fold_y)), max(np.concatenate(all_fold_y))], 'r--', lw=2, label='Ideal')
    plt.title('True vs Predicted Values Across Folds')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    mean_loss = float(np.mean(fold_losses))
    std_loss  = float(np.std(fold_losses))
    mean_r2 = float(np.mean(fold_r2s))
    std_r2 = float(np.std(fold_r2s))
    return mean_loss, std_loss, mean_r2, std_r2
>>>>>>> 21d2d4ebad9e2f27324e55a07035d42c66b47fb9:src/cross_validation_trainer.py
