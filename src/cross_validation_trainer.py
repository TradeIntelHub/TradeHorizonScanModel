from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from model import TradeHorizonScanModel
from torcheval.metrics import R2Score
from typing import List
import numpy as np

global all_the_results
def cross_validate(
    dataset,
    hs_map_size: int,
    #yr_map_size: int,
    dim_trade: int,
    dim_exp: int,
    dim_imp: int,
    dim_cty: int,
    k_splits: int = 5 ,
    batch_size: int = 2048, 
    lr: float = 1e-3,
    epochs: int = 5,#200,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
): 
    global all_the_results
    kfold = KFold(n_splits=k_splits, shuffle=True, random_state=42)
    fold_losses = []
    fold_r2s = []
    all_fold_y = []  # store all y values for each fold
    all_fold_preds = [] # store all predictions for each fold
    all_fold_apes = []
    all_the_results = {} #global dictionary to store the results

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), 1): #_
        print(f"K-Fold {fold} starts")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=4)
        
        #model set up
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

        #train the model
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
            fold_y = []
            fold_preds = [] # store y and preds for each fold 
            fold_apes = [] # store APE for each fold 
            for h_idx, tx, ex, im, ct, y in val_loader:#remove y_idx to match with the model
                h_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx, tx,ex,im,ct,y)]#remove y_idx to match with the model
                preds = model(h_idx, tx, ex, im, ct)
                _ = r2_metric.update(preds, y)
                val_loss += criterion(preds, y).item() * y.size(0)

                actual = y.cpu().numpy()
                predicted = preds.cpu().numpy()
                # calculate MAPE for each fold and aviod division by zero
                mask = actual > 0
                actual = actual[mask]
                predicted = predicted[mask]
                if len(actual) > 0:
                    ape = np.abs((actual - predicted) / actual) * 100
                    fold_apes.append(ape)

                fold_y.append(y.cpu().numpy())  # colect y values for each fold
                fold_preds.append(preds.cpu().numpy()) #collect preds for each fold
        val_loss /= len(val_idx)
        fold_losses.append(val_loss)
        #fold_r2s.append(r2_metric.compute())
        fold_r2s.append(r2_metric.compute().cpu().item())
        fold_y = np.concatenate(fold_y, axis=0)  # 
        fold_preds = np.concatenate(fold_preds, axis=0)  # collect all y and preds for each fold
        all_fold_y.append(fold_y)  # store y values in all_fold_y  for each fold
        all_fold_preds.append(fold_preds) # store preds in all_fold_preds for each fold
        print(f"K-Fold {fold} result: (val_loss: {val_loss:.4f}, r2: {r2_metric.compute().item():.4f})") # print the result of each fold
        #average percentage error for each fold       
        if fold_apes:
                fold_apes = np.concatenate(fold_apes, axis=0)
                mape = np.mean(fold_apes)
                print(f"Fold {fold} MAPE: {mape:.2f}%")
                all_fold_apes.append(fold_apes)
        else:
                print(f"Warning: Fold {fold} has no valid actual values (> 0), skipping MAPE calculation.")
                fold_apes = np.array([])
                mape = np.nan
                all_fold_apes.append(fold_apes)    
    
    #set up variables
    all_y = np.concatenate(all_fold_y, axis=0)
    all_preds = np.concatenate(all_fold_preds, axis=0)
    mean_mse = float(np.mean(fold_losses)) #mean_loss
    std_mse  = float(np.std(fold_losses)) #std_loss
    mean_r2 = float(np.mean(fold_r2s))
    std_r2 = float(np.std(fold_r2s))
    fold_mapes = [] #mean ape for each fold
    for i, apes in enumerate(all_fold_apes):
        if len(apes) > 0:
            mape = np.mean(apes)
            fold_mapes.append(mape)
        else:
            fold_mapes.append(np.nan)
    mean_mape = np.nanmean(fold_mapes)
    print(f" MAPE of this Model: {mean_mape:.2f}%")
    #store in to golbal dictionary
    all_the_results["all_y"] = all_y
    all_the_results["all_preds"] = all_preds
    all_the_results["mean_mse"] = mean_mse
    all_the_results["std_mse"] = std_mse
    all_the_results["mean_r2"] = mean_r2
    all_the_results["std_r2"] = std_r2
    all_the_results["all_fold_apes"] = all_fold_apes
    all_the_results["fold_mapes"] = fold_mapes
    all_the_results["mean_mape"] = mean_mape
    all_the_results["fold_losses"] = fold_losses
    all_the_results["fold_r2s"] = fold_r2s
    print("All the results:", all_the_results)
