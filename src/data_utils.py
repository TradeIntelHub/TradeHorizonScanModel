from typing import Dict, Tuple, Sequence
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

ExporterKey = Tuple[int, int]
ImporterKey = Tuple[int, int]
CountryKey = Tuple[int, int]

def load_maps(
    exporter_path: str,
    importer_path: str,
    country_path: str
) -> Tuple[
    Dict[ExporterKey, np.ndarray],
    Dict[ImporterKey, np.ndarray],
    Dict[CountryKey, np.ndarray]
]:
 # read and build raw maps
    exporter_df = pd.read_csv(exporter_path)
    raw_exp = {
        (row.exporter, row.year): np.array([
            row['MA_Theil_Exporter_Concentration'],
            row['MA_GDPPerCapita_exporter'],
            row['MA_GeopoliticalIndex_exporter'],
            row['MA_ConsumerPriceIndex_exporter']
        ], dtype=np.float32)
        for _, row in exporter_df.iterrows()
    }
    importer_df = pd.read_csv(importer_path)
    raw_imp = {
        (row.importer, row.year): np.array([
            row['MA_Theil_Importer_Concentration'],
            row['MA_GDPPerCapita_importer'],
            row['MA_TariffRatesAllProductsWeigthedAverage_importer'],
            row['MA_GeopoliticalIndex_importer'],
            row['MA_ConsumerPriceIndex_importer']
        ], dtype=np.float32)
        for _, row in importer_df.iterrows()
    }
    country_df = pd.read_csv(country_path)
    raw_cty = {
        (row.importer, row.exporter): np.array([
            row['MA_contig'],
            row['MA_dist'],
            row['MA_landlocked']
        ], dtype=np.float32)
        for _, row in country_df.iterrows()
    }

    # standardize each branch's features
    def standardize(raw_map: Dict) -> Dict:
        arr = np.stack(list(raw_map.values()), axis=0)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        return {k: (v - mean) / std for k, v in raw_map.items()}

    exp_map = standardize(raw_exp)
    imp_map = standardize(raw_imp)
    cty_map = standardize(raw_cty)

    return exp_map, imp_map, cty_map

class TradeDataset(Dataset):
    def __init__(
        self,
        trd_path: str,
        exp_map: Dict[ExporterKey, np.ndarray],
        imp_map: Dict[ImporterKey, np.ndarray],
        cty_map: Dict[CountryKey, np.ndarray],
        trd_feats: Sequence[str]
    ) -> None:
        df = pd.read_csv(
            trd_path,
            usecols=[*['importer', 'exporter', 'hsCode', 'year', 'MA_value'], *trd_feats]
        )
        self.df = df.reset_index(drop=True)
        self.trd_feats = list(trd_feats)
        # compute mean/std for trade features
        feat_arr = self.df[self.trd_feats].to_numpy(dtype=np.float32)
        self.trd_mean = feat_arr.mean(axis=0)
        self.trd_std  = feat_arr.std(axis=0)
        # categorical maps
        self.hs_map = {code: i for i, code in enumerate(self.df['hsCode'].unique())}
        # For now we don't use year as a feature, so we don't need to map it to an index
        # self.yr_map = {yr: i for i, yr in enumerate(self.df['year'].unique())}
        # branch maps (already standardized)
        self.exp_map = exp_map
        self.imp_map = imp_map
        self.cty_map = cty_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        # embeddings
        h_idx = self.hs_map[row.hsCode]
        # y_idx = self.yr_map[row.year]
        # trade x standardized
        raw_trade = np.array([row[col] for col in self.trd_feats], dtype=np.float32)
        std_trade = (raw_trade - self.trd_mean) / self.trd_std
        trd_x = torch.tensor(std_trade, dtype=torch.float32)
        # other branches (already standardized in maps)
        exp_x = torch.tensor(self.exp_map[(row.exporter, row.year)], dtype=torch.float32)
        imp_x = torch.tensor(self.imp_map[(row.importer, row.year)], dtype=torch.float32)
        cty_x = torch.tensor(self.cty_map[(row.importer, row.exporter)], dtype=torch.float32)
        target = torch.tensor(row.MA_value, dtype=torch.float32)

        return (
            torch.tensor(h_idx, dtype=torch.long),
            # torch.tensor(y_idx, dtype=torch.long),
            trd_x,
            exp_x,
            imp_x,
            cty_x,
            target
        )
