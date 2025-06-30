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
import plotly.graph_objects as go
import plotly.express as px
from torcheval.metrics import R2Score

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
train_idx, val_idx = train_test_split(np.arange(len(dataset.df)), test_size=0.2, random_state=7, shuffle=True)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=8)



# Setting the final training parameters
early_stopping = EarlyStopping(patience=100, delta=0)
epochs = 40
all_train_losses = []
all_val_losses = []

# Training the model
start_time = time.time()
for epoch in range(epochs):
    _ = model.train()
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
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    # validation step
    _ = model.eval()
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

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:,.2f}, Val Loss: {val_loss:,.2f}')
    end_time = time.time()
    print(f'Training time so far (epoch: {epoch+1}): {(end_time - start_time)/3600:.2f} hours')
    print(f'Estimated time remaining: {((end_time - start_time)/3600) /(epoch+1) * (epochs - epoch - 1):.2f} hours')
    

    early_stopping(val_loss, model, all_val_losses)
    if early_stopping.is_this_best_so_far:
        torch.save({'epoch': len(all_train_losses),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'all_train_losses': all_train_losses,
        'all_val_losses': all_val_losses,
                }, f'../TradeHorizonScan/models/best_model.pth')
        print(f'Best model saved at epoch {epoch+1}')
        
    print('=='* 50)
    if early_stopping.early_stop:
        print("Early stopping")
        break







####################################################
# There are nans in the dataset.Alberta_df
# For Harmful and Liberalising
# Try to fix them using the original GTA dataset
# IF there are still nans, then fill them with 0
####################################################



'''
# Loading the best model
checkpoint = torch.load('../TradeHorizonScan/models/checkpoint243.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
all_train_losses = checkpoint['all_train_losses']
all_val_losses = checkpoint['all_val_losses']
_ = model.eval()
'''

# Calculate the R squared value
r2_metric = R2Score().to(device) 
actuals = []
predictions = []
with torch.no_grad():
    for h_idx, tx, ex, im, ct, y in val_loader:
        h_idx, tx, ex, im, ct, y = [t.to(device) for t in (h_idx, tx, ex, im, ct, y)]
        preds = model(h_idx, tx, ex, im, ct)
        actuals.extend(y.cpu().numpy())
        predictions.extend(preds.cpu().numpy())
        _ = r2_metric.update(preds, y)
r2 = r2_metric.compute()
print(f"R-squared: {r2.item():.2f}")
print(min(all_val_losses)**0.5)
print(min(all_val_losses)**0.5/dataset.df['MA_value'].mean())

# Plotting the training and validation loss
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(1, len(all_train_losses) + 1)), 
    y=all_train_losses,
    mode='lines+markers',
    name='Training Loss',
    marker=dict(size=8, symbol='circle'),
    line=dict(width=2, color='#1f77b4'),
    hovertemplate='Epoch: %{x}<br>Loss: %{y:,.0f}<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=list(range(1, len(all_val_losses) + 1)), 
    y=all_val_losses,
    mode='lines+markers',
    name='Validation Loss',
    marker=dict(size=8, symbol='diamond'),
    line=dict(width=2, color='#ff7f0e', dash='dash'),
    hovertemplate='Epoch: %{x}<br>Loss: %{y:,.0f}<extra></extra>'
))
fig.update_layout(
    template='plotly_white',
    title={
        'text': 'Model Training and Validation Loss',
        'font': {'size': 24, 'family': 'Arial', 'color': '#333333'},
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1,
        font=dict(size=12)
    ),
    xaxis=dict(
        title='Epochs',
        titlefont=dict(size=14, family='Arial'),
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E5E5',
        zeroline=False,
        dtick=10,
        range=[-9, None] 
    ),
    yaxis=dict(
        title='Loss',
        titlefont=dict(size=14, family='Arial'),
        showgrid=True,
        gridwidth=1,
        gridcolor='#E5E5E5',
        zeroline=True
    ),
    width=1200,
    height=700,
    margin=dict(l=80, r=80, t=100, b=80),
    annotations=[
        dict(
            text=f'Learning Rate: {lr}, Batch Size: {batch_size}',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.5,
            y=-0.13,
            font=dict(size=12)
        )
    ]
)
fig.show()


# Scatter plot of actual vs predicted values
min_val = min(min(actuals), min(predictions))
max_val = max(max(actuals), max(predictions))
fig = px.scatter(
    x=actuals,
    y=predictions,
    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
    template='plotly_white'
)
fig.update_traces(
    marker=dict(size=5, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
    hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'
)
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    line=dict(color='gray', dash='dash'),
    showlegend=False
))
font_style = dict(family='Arial', size=14, color='#333333')
fig.update_layout(
    xaxis=dict(
        title='Actual Values',
        showgrid=True,
        gridcolor='#E5E5E5',
        zeroline=True,
        range=[0, max(actuals) * 1.05],     
        scaleanchor="y",
        scaleratio=1
    ),
    yaxis=dict(
        title='Predicted Values',
        showgrid=True,
        gridcolor='#E5E5E5',
        zeroline=False,
        range=[0, max(predictions) * 1.05]
    ),
    title={
        'text': 'Actual vs Predicted Values',
        'font': {'size': 24, 'family': 'Arial', 'color': '#333333'},
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    font=dict(family='Arial', size=14, color='#333333'),
    width=1200,
    height=700,
    margin=dict(l=80, r=80, t=100, b=80)
)
fig.update_layout(
    xaxis=dict(scaleanchor="y", scaleratio=1)
)
fig.show()


# MAPE analysis
x = []
y = []
quantiles = [0, 0.25, 0.5, 0.75, 1]
quantiles = np.arange(0, 1.1, 0.1)
for j, i in enumerate(quantiles[1:]):
    q = np.quantile(actuals, i)
    print(f'Quantile {i}')
    if i == quantiles[1]:
        predictions_q = np.array(predictions)[(np.array(actuals) <= q) ]
        actuals_q = np.array(actuals)[(np.array(actuals) <= q) ]
    elif i == quantiles[-1]:
        predictions_q = np.array(predictions)[(np.array(actuals) > actuals_q.max()) ]
        actuals_q = np.array(actuals)[(np.array(actuals) > actuals_q.max()) ]
    else:
        predictions_q = np.array(predictions)[(np.array(actuals) <= q) & (np.array(actuals) > actuals_q.max()) ]
        actuals_q = np.array(actuals)[(np.array(actuals) <= q) & (np.array(actuals) > actuals_q.max()) ]

    mape_q = np.mean(np.abs((predictions_q - actuals_q) / actuals_q)) * 100
    x.append((actuals_q.min(), actuals_q.max()))
    y.append(mape_q)
    print(f'MAPE for {j+1}th quantile: Min Trade Value: {actuals_q.min():,.2f}')
    print(f'MAPE for {j+1}th quantile: Median Trade Value: {np.median(actuals_q):,.2f}')
    print(f'MAPE for {j+1}th quantile: Mean Trade Value: {actuals_q.mean():,.2f}')
    print(f'MAPE for {j+1}th quantile: Max Trade Value: {actuals_q.max():,.2f}')
    print(f'MAPE for {j+1}th quantile: {mape_q:,.2f}%')
    print('=='*50)

bars = [{"x_start": quantiles[i], "x_end": quantiles[i+1], "height": y[i]} for i in range(len(x))]
fig = go.Figure()
for bar in bars:
    fig.add_shape(
        type="rect",
        x0=bar["x_start"], x1=bar["x_end"],
        y0=0, y1=bar["height"],
        line=dict(color="black"),
        fillcolor="royalblue",
    )
fig.update_layout(
    title="Custom Bar Chart with Start/End Positions",
    xaxis=dict(range=[quantiles[0]-0.2, quantiles[-1]+0.2], title="X"),
    yaxis=dict(range=[0, np.array(y).max()+5], title="Height"),
    showlegend=False
)
fig.show()
