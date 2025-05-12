from typing import List
from data_utils import load_maps, TradeDataset
from cross_validation_trainer import cross_validate


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

    mean_mse, std_mse,mean_r2, std_r2,all_fold_apes = cross_validate(
        dataset = dataset,
        hs_map_size = len(dataset.hs_map),
        #yr_map_size = len(dataset.year_map),
        dim_trade = len(trade_feats),
        dim_exp = next(iter(exporter_map.values())).shape[0],
        dim_imp = next(iter(importer_map.values())).shape[0],
        dim_cty = next(iter(country_map.values())).shape[0]
    )
    print(f"MSE: {mean_mse}, {std_mse}")
    print(f"R²: {mean_r2}, {std_r2}")
    print(f"average validation MSE: {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"average validation R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Each Fold's MAPE values: {[ape.mean().round(2) for ape in all_fold_apes]}")
if __name__ == '__main__':
    main()
