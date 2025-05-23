from typing import Dict, Tuple, Sequence
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pyodbc
import itertools

ExporterKey = Tuple[int, int]
ImporterKey = Tuple[int, int]
CountryKey = Tuple[int, int]

conn = pyodbc.connect(
                        'DRIVER={ODBC Driver 17 for SQL Server};'
                        'SERVER=c-goa-sql-10593;'
                        'DATABASE=JET_TRDAT_UAT;'
                        'Trusted_Connection=yes;'
                    )

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
            self.prepare_final_Alberta_df()
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
        self.ctyCode_to_country = dict(zip(Country_codes['ctyCode'], Country_codes['Country']))
    def prepare_final_Alberta_df(self):
        # This function is used to prepare the Alberta dataframe for the final output

        # Adding ALberta and the rest of the world to the country_to_code and code_to_country
        self.country_to_code['Alberta'] = self.Alberta_df['exporter'].unique()[0]
        self.country_to_code['Rest of the World'] = 9998
        self.country_to_code['Canada'] = 124
        self.code_to_country[124] = 'Canada'
        self.code_to_country[self.Alberta_df['exporter'].unique()[0]] = 'Alberta'
        self.code_to_country[9998] = 'Rest of the World'



        unique_importer_codes = list(self.code_to_country.keys())
        unique_importer_codes.remove(9999)
        unique_importer_codes.remove(124)
        unique_years = [self.df['year'].max()]
        unique_hs_codes = self.df['hsCode'].unique()

        Alberta_full_trade_matrix = pd.DataFrame(list(itertools.product(unique_years, unique_importer_codes, unique_hs_codes)), 
                                                columns=['year', 'importer', 'hsCode'])
        Alberta_full_trade_matrix['exporter'] = self.Alberta_df.exporter.unique()[0]
        print(f'There are {len(Alberta_full_trade_matrix):,} possible trade flows for Alberta ({len(unique_hs_codes)} HS codes, {unique_years[0]} year, {len(unique_importer_codes)} importers)')




        cols_on = ['year']
        cols_to_merge = ['Covid']
        cols_to_merge.extend(cols_on)
        Alberta_full_trade_matrix = pd.merge(Alberta_full_trade_matrix, self.df[cols_to_merge].drop_duplicates(), on=cols_on, how='left')

        cols_on = ['year', 'hsCode']
        cols_to_merge = ['MA_Harmful', 'MA_Liberalising'] 
        cols_to_merge.extend(cols_on)
        Alberta_full_trade_matrix = pd.merge(Alberta_full_trade_matrix, self.df[cols_to_merge].drop_duplicates(), on=cols_on, how='left')

        cols_on = ['year', 'hsCode', 'exporter'] # NOtice that if exporter is one of the key columns, then we need to use the Alberta_df
        cols_to_merge = ['MA_Partner_Revealed_Comparative_Advantage', 'MA_TotalExportofCmdbyPartner', 'MA_AvgUnitPriceofExporterToWorldFlags', 'MA_AvgUnitPriceofExporterToWorld'] 
        cols_to_merge.extend(cols_on)
        Alberta_full_trade_matrix = pd.merge(Alberta_full_trade_matrix, self.Alberta_df[cols_to_merge].drop_duplicates(), on=cols_on, how='left')

        cols_on = ['year', 'hsCode', 'exporter', 'importer']
        cols_to_merge = ['MA_Trade_Complementarity'] 
        cols_to_merge.extend(cols_on)
        Alberta_full_trade_matrix = pd.merge(Alberta_full_trade_matrix, self.Alberta_df[cols_to_merge].drop_duplicates(), on=cols_on, how='left')

        cols_on = ['year', 'hsCode', 'importer']
        cols_to_merge = ['MA_TotalImportofCmdbyReporter', 'MA_AvgUnitPriceofImporterFromWorldFlags', 'MA_AvgUnitPriceofImporterFromWorld'] 
        cols_to_merge.extend(cols_on)
        Alberta_full_trade_matrix = pd.merge(Alberta_full_trade_matrix, self.df[cols_to_merge].drop_duplicates(), on=cols_on, how='left')

        cols_on = ['year', 'hsCode', 'importer', 'exporter']
        cols_to_merge = ['MA_AvgUnitPrice', 'MA_value', 'MA_AvgUnitPriceFlags'] 
        cols_to_merge.extend(cols_on)
        Alberta_full_trade_matrix = pd.merge(Alberta_full_trade_matrix, self.Alberta_df[cols_to_merge].drop_duplicates(), on=cols_on, how='left')



        # Getting the real trade data for hte Alberta in 2024 (prediction year)
        latest_year = 2024
        Alberta_Actual_exports = pd.read_sql(f"SELECT * FROM dbo.statCanExportDataMonthly{latest_year}", conn)
        Alberta_Actual_exports = Alberta_Actual_exports.loc[Alberta_Actual_exports['provCode'] == 'AB'].reset_index(drop=True)
        Alberta_Actual_exports.ctyCode = Alberta_Actual_exports.ctyCode.replace("TW", "CN")
        Alberta_Actual_exports['hsCode'] = Alberta_Actual_exports['hs6Code'].astype(str).str.zfill(6).str.slice(0, 4).astype(int)
        Alberta_Actual_exports['year'] = latest_year
        Alberta_Actual_exports['exporter'] = self.Alberta_df.exporter.unique()[0]

        Alberta_Actual_exports['importer'] = Alberta_Actual_exports['ctyCode'].map(self.ctyCode_to_country).map(self.country_to_code)
        Alberta_Actual_exports.loc[Alberta_Actual_exports.importer.isna(), 'importer'] = 9998
        Alberta_Actual_exports['importer'] = Alberta_Actual_exports['importer'].astype(int)
        Alberta_Actual_exports.drop(columns=['stateCode', 'Quantity', 'provCode', 'hs6Code', 'YearMonth', 'ctyCode'], inplace=True)
        Alberta_Actual_exports = Alberta_Actual_exports.groupby(['hsCode', 'year', 'importer', 'exporter'], as_index=False).agg({'Value': 'sum'})
        Alberta_Actual_exports['Value'] = Alberta_Actual_exports['Value']/1000
        print(f'Alberta had {len(Alberta_Actual_exports):,} actual trade flows ({len(Alberta_Actual_exports.hsCode.unique())} HS codes, {Alberta_Actual_exports.year.unique()[0]} year, {len(Alberta_Actual_exports.importer.unique())} importers)')
        Alberta_Actual_exports.rename(columns={'Value': f'Actual_Alberta_{latest_year}_Values'}, inplace=True)
        # Merging the Alberta trade data with the Alberta full trade matrix
        Alberta_full_trade_matrix = pd.merge(Alberta_full_trade_matrix, Alberta_Actual_exports.drop('year', axis=1), on=['importer', 'exporter','hsCode'], how='left')
        RoW = Alberta_full_trade_matrix.loc[Alberta_full_trade_matrix.importer == 9998]
        RoW = RoW[RoW.Actual_Alberta_2024_Values>0]
        RoW = RoW[['hsCode', 'Actual_Alberta_2024_Values']]
        self.rest_of_the_world_hsCode_to_Trade_Value_2024 = dict(zip(RoW['hsCode'], RoW['Actual_Alberta_2024_Values']))
        Alberta_full_trade_matrix = Alberta_full_trade_matrix.loc[(Alberta_full_trade_matrix.importer != 9998)]
        print(f'In {latest_year}, Alberta has only traded {len(Alberta_Actual_exports.hsCode.unique())} unique HS4 codes out of the {len(Alberta_full_trade_matrix.hsCode.unique())} HS4 codes.')
        l = len(Alberta_full_trade_matrix)
        Alberta_full_trade_matrix = Alberta_full_trade_matrix.loc[Alberta_full_trade_matrix.hsCode.isin(Alberta_Actual_exports.hsCode.unique())]
        print(f'As a result, out of the {l:,} possible trade flows, Alberta only has the potential of {len(Alberta_full_trade_matrix):,} trade flows.')
        Alberta_full_trade_matrix =  Alberta_full_trade_matrix.loc[:, list(self.Alberta_df.columns) + ['Actual_Alberta_2024_Values']]
        l = len(Alberta_full_trade_matrix)
        Alberta_full_trade_matrix = Alberta_full_trade_matrix.loc[~Alberta_full_trade_matrix.MA_TotalImportofCmdbyReporter.isna()]
        print(f'Out of the {l:,} possible trade flows, There is only demand by importer in {len(Alberta_full_trade_matrix):,} trade flows.')
        print(f'The database still has many nan values for the trades that did not happen. As a first step, I use the Canada numbers to fill in features with missing values for Alberta')
        print(f'{(Alberta_full_trade_matrix.isna().sum(axis=1)>0).sum():,} rows with Nans out of {len(Alberta_full_trade_matrix):,} possible trade flows')

        # NO trade happened , so value is 0
        Alberta_full_trade_matrix.loc[Alberta_full_trade_matrix.MA_value.isna(), 'MA_value'] = 0
        Alberta_full_trade_matrix.loc[Alberta_full_trade_matrix.Actual_Alberta_2024_Values.isna(), 'Actual_Alberta_2024_Values'] = 0


        cols_on = ['year', 'hsCode'] 
        cols_to_merge = ['MA_Partner_Revealed_Comparative_Advantage', 'MA_TotalExportofCmdbyPartner', 'MA_AvgUnitPriceofExporterToWorldFlags', 'MA_AvgUnitPriceofExporterToWorld'] 
        cols_to_merge.extend(cols_on)
        Alberta_full_trade_matrix = pd.merge(Alberta_full_trade_matrix, 
                                            self.df.loc[self.df.exporter==124, cols_to_merge].drop_duplicates(), on=cols_on, how='left', suffixes=('', '_canada'))
        cols_to_merge = ['MA_Partner_Revealed_Comparative_Advantage', 'MA_TotalExportofCmdbyPartner', 'MA_AvgUnitPriceofExporterToWorldFlags', 'MA_AvgUnitPriceofExporterToWorld'] 
        for col in cols_to_merge:
            Alberta_full_trade_matrix[col] = Alberta_full_trade_matrix[col].combine_first(Alberta_full_trade_matrix[f'{col}_canada'])
            Alberta_full_trade_matrix.drop(f'{col}_canada', axis=1, inplace=True)

        cols_on = ['year', 'hsCode', 'importer'] 
        cols_to_merge = ['MA_AvgUnitPrice', 'MA_AvgUnitPriceFlags', 'MA_Trade_Complementarity'] 
        cols_to_merge.extend(cols_on)
        Alberta_full_trade_matrix = pd.merge(Alberta_full_trade_matrix, 
                                            self.df.loc[self.df.exporter==124, cols_to_merge].drop_duplicates(), on=cols_on, how='left', suffixes=('', '_canada'))
        cols_to_merge = ['MA_AvgUnitPrice', 'MA_AvgUnitPriceFlags', 'MA_Trade_Complementarity']
        for col in cols_to_merge:
            Alberta_full_trade_matrix[col] = Alberta_full_trade_matrix[col].combine_first(Alberta_full_trade_matrix[f'{col}_canada'])
            Alberta_full_trade_matrix.drop(f'{col}_canada', axis=1, inplace=True)
        print('After filling in the missing values with the Canada numbers, we still have some missing values related to the average unit price of the trade which will be resolved by flagging them')
        print(f'{(Alberta_full_trade_matrix.isna().sum(axis=1)>0).sum():,} rows with Nans out of {len(Alberta_full_trade_matrix):,} possible trade flows')
        Alberta_full_trade_matrix.MA_AvgUnitPrice = Alberta_full_trade_matrix.MA_AvgUnitPrice.fillna(0)
        Alberta_full_trade_matrix.loc[Alberta_full_trade_matrix.MA_AvgUnitPrice == 0, 'MA_AvgUnitPriceFlags'] = True
        print('Now the only variable with Nans is the MA_Trade_Complementarity which I am going to fill with the mean of the column (for 30 countries)')
        Alberta_full_trade_matrix.MA_Trade_Complementarity = Alberta_full_trade_matrix.MA_Trade_Complementarity.fillna(self.df.MA_Trade_Complementarity.mean())
        self.Alberta_df = Alberta_full_trade_matrix
        print(f'Now we have {len(self.Alberta_df):,} trade flows (Actual and Potential) with no missing values')
        return






'''
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
dataset.Alberta_df
'''




