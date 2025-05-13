from typing import List
from src.data_utils import load_maps, TradeDataset
from src.cross_validation_trainer import cross_validate
import matplotlib.pyplot as plt
import numpy as np


global all_the_results #access the global dictionary
all_the_results = {}

def main() -> None:
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


    dataset.df = dataset.df.sample(frac=0.001, random_state=42).reset_index(drop=True)
    print(f"Dataset created, size: {len(dataset)}")

    cross_validate(
        dataset=dataset,
        hs_map_size=len(dataset.hs_map),
        dim_trade=len(trade_feats),
        dim_exp=next(iter(exporter_map.values())).shape[0],
        dim_imp=next(iter(importer_map.values())).shape[0],
        dim_cty=next(iter(country_map.values())).shape[0]
    )


    mean_mse = all_the_results["mean_mse"]
    std_mse = all_the_results["std_mse"]
    mean_r2 = all_the_results["mean_r2"]
    std_r2 = all_the_results["std_r2"]
    all_fold_apes = all_the_results["all_fold_apes"]
    print(f"MSE: {mean_mse}, {std_mse}")
    print(f"R²: {mean_r2}, {std_r2}")
    print(f"average validation MSE: {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"average validation R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Each Fold's MAPE values: {[ape.mean().round(2) for ape in all_fold_apes]}")

if __name__ == '__main__':
    main()

#plot
#plot the loss in rmse
fold_losses = all_the_results["fold_losses"]
rmse = np.sqrt(fold_losses) #calculate RMSE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
folds = list(range(1, len(fold_losses) + 1))
plt.plot(folds, rmse, marker='o', linestyle='-', color='b', label='Validation Loss')
plt.title('Validation Loss (MSE) Across Folds')
plt.xlabel('Fold')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.xticks(folds, [int(f) for f in folds])
plt.grid(True)
plt.legend()

# Print R² Score
fold_r2s = all_the_results["fold_r2s"]

plt.subplot(1, 2, 2)
plt.plot(folds, fold_r2s, marker='o', linestyle='-', color='g', label='R² Score')
plt.title('R² Score Across Folds')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.xticks(folds, [int(f) for f in folds])
plt.grid(True)
plt.legend()

plt.tight_layout()
#plt.savefig('mse_r2_plot.png')   
plt.show() 


#plot all y and preds 
all_y = all_the_results["all_y"]
all_preds = all_the_results["all_preds"]
scale_factor = 1e6  # 1 million
all_y_scaled = all_y / scale_factor
all_preds_scaled = all_preds / scale_factor


plt.figure(figsize=(6, 5))
plt.scatter(all_y_scaled, all_preds_scaled, alpha=0.5, s=10, label='Predicted vs True')
plt.plot([min(all_y_scaled), max(all_y_scaled)], [min(all_y_scaled), max(all_y_scaled)], 'r--', lw=2, label='Ideal')
plt.title('True vs Predicted Values (All Folds Combined)')
plt.xlabel('True Values (Millions $)')
plt.ylabel('Predicted Values (Millions $)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('true_vs_pred_plot.png')

plt.show() 
