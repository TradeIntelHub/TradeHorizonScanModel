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
    #dataset.df = dataset.df.sample(frac=0.001, random_state=42).reset_index(drop=True)
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

#scatter plot
#plot all y and preds 
actuals = all_the_results["all_y"]
predictions = all_the_results["all_preds"]
scale_factor = 1e6  # 1 million
actuals_scaled = actuals / scale_factor
predictions_scaled = predictions / scale_factor


plt.figure(figsize=(6, 5))
plt.scatter(actuals_scaled, predictions_scaled, alpha=0.5, s=10, label='Predicted vs True')
plt.plot([min(actuals_scaled), max(actuals_scaled)], [min(actuals_scaled), max(actuals_scaled)], 'r--', lw=2, label='Ideal')
plt.title('True vs Predicted Values (All Folds Combined)')
plt.xlabel('True Values (Millions $)')
plt.ylabel('Predicted Values (Millions $)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('true_vs_pred_plot.png')

plt.show() 


mean_mse = all_the_results["mean_mse"]
std_mse = all_the_results["std_mse"]
std_rmse= np.sqrt(std_mse)
mean_rmse = np.sqrt(mean_mse)
print(f"RMSE: {mean_rmse:.4f}, {std_rmse:.4f}")