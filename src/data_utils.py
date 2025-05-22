from typing import Dict, Tuple, Sequence
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pyodbc

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
    # Replacing NaN values with mean for the columns
    exporter_df["MA_GeopoliticalIndex_exporter"] = exporter_df["MA_GeopoliticalIndex_exporter"].fillna(exporter_df["MA_GeopoliticalIndex_exporter"].mean())
    exporter_df["MA_TariffRatesAllProductsWeigthedAverage_exporter"] = exporter_df["MA_TariffRatesAllProductsWeigthedAverage_exporter"].fillna(exporter_df["MA_TariffRatesAllProductsWeigthedAverage_exporter"].mean())
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
    # Replacing NaN values with mean for the columns
    importer_df["MA_GeopoliticalIndex_importer"] = importer_df["MA_GeopoliticalIndex_importer"].fillna(importer_df["MA_GeopoliticalIndex_importer"].mean())
    importer_df["MA_TariffRatesAllProductsWeigthedAverage_importer"] = importer_df["MA_TariffRatesAllProductsWeigthedAverage_importer"].fillna(importer_df["MA_TariffRatesAllProductsWeigthedAverage_importer"].mean())
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
            row['MA_dist']], dtype=np.float32)
        for _, row in country_df.iterrows()
    }

    # standardize each branch's features
    def standardize(raw_map: Dict) -> Dict:
        arr = np.stack(list(raw_map.values()), axis=0)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
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
        trd_feats: Sequence[str],
        **kwargs
    ) -> None:
        """
        Keyword arguments:
        - inferece_mode: bool: if True, load Alberta data for inference. Default False.
        - Alberta_path: str: path to Alberta data. Required if inference_mode is True.
        """
        df = pd.read_csv(
            trd_path,
            usecols=[*['importer', 'exporter', 'hsCode', 'year', 'MA_value'], *trd_feats]
        )
        self.df = df.reset_index(drop=True)
        self.inference_mode = kwargs.get('inference_mode', False)
        if self.inference_mode:
            Alberta_df = pd.read_csv(
                kwargs['Alberta_path'],
                usecols=[*['importer', 'exporter', 'hsCode', 'year', 'MA_value'], *trd_feats]
            )
            self.Alberta_df = Alberta_df
            self.Alberta_df = Alberta_df.reset_index(drop=True)
            self.Unify_Country_Codes()
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
        if self.inference_mode:
            # In inference mode, we use Alberta data
            row = self.Alberta_df.iloc[idx]
        else:   
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

    def Unify_Country_Codes(self):
        conn = pyodbc.connect(
                                'DRIVER={ODBC Driver 17 for SQL Server};'
                                'SERVER=c-goa-sql-10593;'
                                'DATABASE=JET_TRDAT_UAT;'
                                'Trusted_Connection=yes;'
                            )
        # define a function where we get the actual name of the cocuntries if code is given and we get the code if the name is given
        Country_codes = pd.read_sql("SELECT * FROM dbo.countryCodesDescriptionStatCanUNComTradeUSCensusBureau", conn)
        Country_codes = Country_codes.loc[:, ['Country', 'ctyCode', 'UNComTradeCtyId']]
        Country_codes = Country_codes.loc[Country_codes['UNComTradeCtyId'].isin(set(self.df['importer'].unique()).union(self.Alberta_df['importer'].unique()))].reset_index(drop=True)
        Country_codes.to_csv("data\\Country_codes.csv", index=False)
        code_to_country = dict(zip(Country_codes['UNComTradeCtyId'], Country_codes['Country']))
        a = Country_codes.loc[Country_codes['UNComTradeCtyId'].isin(self.df['importer'].unique())]
        country_to_code = dict(zip(a['Country'], a['UNComTradeCtyId']))
        self.Alberta_df['importer'] = self.Alberta_df['importer'].map(code_to_country).map(country_to_code)
        if self.Alberta_df['importer'].isna().any():
            missing_codes = self.Alberta_df['importer'].loc[self.Alberta_df.importer.isna()].unique()
            raise ValueError(f"Unmapped country codes found: {missing_codes}")
        Country_codes = Country_codes.loc[Country_codes.UNComTradeCtyId.isin(self.Alberta_df['importer'].unique())].reset_index(drop=True)
        Country_codes.Country = Country_codes.Country.replace('Viet Nam', 'Vietnam')
        Country_codes.Country = Country_codes.Country.replace('China', 'China & Taiwan')
        self.code_to_country = dict(zip(Country_codes['UNComTradeCtyId'], Country_codes['Country']))
        self.country_to_code = dict(zip(Country_codes['Country'], Country_codes['UNComTradeCtyId']))








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
    trd_feats = trade_feats,
    inference_mode = True,
    Alberta_path = '../TradeHorizonScan/data/MA_Trade_Alberta.csv'
)


