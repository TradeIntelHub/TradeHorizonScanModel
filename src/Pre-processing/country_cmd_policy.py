import os
import pandas as pd
import numpy as np
start_year = 2012
end_year = 2023



# Horizontal Interventions: Means it possibly affects all products and sectors
# Having access to variables such as 'Distorted market', and 'Affected commercial flows' could provide more accurate representation of the intervention's impact.
# Here is a link to tghe GTA full documentation: https://sgeptupload.s3.eu-west-1.amazonaws.com/gta/GTA+handbook.pdf
# Also I have downloaded the GTA data from this link: https://globaltradealert.org/data-center
# Lastly, I believe dedicating enought time there is room for improvement in the data cleaning process, especially in terms of handling missing values and filtering the data, etc.




data= pd.read_csv('../TradeHorizonScan/src/Pre-processing/data/Trade_Policies_GTA.csv')
data.columns

cols = ['Intervention ID', 'GTA Evaluation', 'Implementing Jurisdictions',
        'Intervention Type', 'Affected Sectors', 'Affected Products',
       'Affected Jurisdictions', 'Date Announced',
       'Date Implemented', 'Date Removed', 'Is In Force', 'Is Horizontal']
data = data.loc[:, cols]


# Taking care of the Implementing Jurisdictions
data['Implementing Jurisdictions'] = data['Implementing Jurisdictions'].str.split(',')
data['Implementing Jurisdictions'] = data['Implementing Jurisdictions'].apply(lambda lst: [c.strip() for c in lst])  
data = data.explode('Implementing Jurisdictions').reset_index(drop=True)


# Taking care of the Affected Sectors
data['Affected Jurisdictions'].replace(np.nan, 'all', inplace=True) # I assume if there is no affected jurisdiction, it means it affects all jurisdictions
data['Affected Jurisdictions'] = data['Affected Jurisdictions'].str.split(',')
data['Affected Jurisdictions'] = data['Affected Jurisdictions'].apply(lambda lst: [c.strip() for c in lst])  
data = data.explode('Affected Jurisdictions').reset_index(drop=True)


all_country_names_GAT = list(data['Implementing Jurisdictions'].unique())
all_country_names_GAT.extend(list(data['Affected Jurisdictions'].unique()))
all_country_names_GAT = set(list(all_country_names_GAT))  # Remove duplicates

countries_to_keep = pd.read_csv('../TradeHorizonScan/src/Pre-processing/data/country_list.csv')
countries_to_keep = list(countries_to_keep['Country'].values)
countries_to_keep.extend(['all'])
# Taking care of name discrepancies
for country in all_country_names_GAT:
    if country in countries_to_keep:
        continue
    else:
        print(country)

# Chaning the GTA country names to match the rest of the data which is recorded in the countries_to_keep 
# Republic of Korea ---> Korea, South
# Vietnam ---> Viet Nam
data.replace(['Republic of Korea'], ['Korea, South'], inplace=True)
data.replace(['Vietnam'], ['Viet Nam'], inplace=True)
data.replace(['Chinese Taipei'], ['China'], inplace=True)

all_country_names_GAT = list(data['Implementing Jurisdictions'].unique())
all_country_names_GAT.extend(list(data['Affected Jurisdictions'].unique()))
all_country_names_GAT = set(list(all_country_names_GAT))  # Remove duplicates

for country in countries_to_keep:
    if country in all_country_names_GAT:
        continue
    elif country == 'Taiwan':
        continue # Taiwan is not in the GTA data so I have combined China and Taiwas, but it is in the countries_to_keep list
    else:
        raise ValueError(f"Country {country} is not in the GTA data. Please check the country names in the GTA data.")



data = data.loc[(data['Implementing Jurisdictions'].isin(countries_to_keep)) & (data['Affected Jurisdictions'].isin(countries_to_keep))]
data.reset_index(drop=True, inplace=True)



# Handiling the date
data['Date'] = data['Date Implemented'].fillna(data['Date Implemented'])
# I assume if neither of the date values are present then the policy has been implemented for the whole period of the GTA data
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Date Removed'] = pd.to_datetime(data['Date Removed'], errors='coerce')
data['Date'] = data['Date'].fillna(pd.Timestamp('2012-01-01'))  # Fill NaT with the start of our period
data['Date Removed'] = data['Date Removed'].fillna(pd.Timestamp('2023-12-20'))  # Fill NaT with the end of our period
data = data.loc[data['Date'].dt.year <= end_year] # Implemented after 2023 then removed
data = data.loc[data['Date Removed'].dt.year >= start_year] # Policy Removed before 2012 then remove it
data['Date'] = data['Date'].dt.year
data['Date Removed'] = data['Date Removed'].dt.year
data['years'] = data.apply(lambda row: list(range(row['Date'], row['Date Removed'] + 1)), axis=1)


# I only keep Red and Green interventions, which are harmful and liberalising respectively
data = data.loc[data['GTA Evaluation'].isin(['Red', 'Green'])].reset_index(drop=True)




# Final columns to keep
cols = ['Implementing Jurisdictions', 'GTA Evaluation', 'Affected Jurisdictions', 'Affected Products','years']
data = data.loc[:, cols]


# Taking care of the Affected Products
data['Affected Products'].replace(np.nan, 'all', inplace=True)
data['Affected Products'] = data['Affected Products'].str.split(',') 
data['Affected Products'] = data['Affected Products'].apply(lambda lst: [c.strip() for c in lst])



# Explode the Affected Products and the Years columns
print(f'{len(data):,}')
data = data.explode('Affected Products').reset_index(drop=True)
print(f'{len(data):,}')
data = data.explode('years').reset_index(drop=True)
print(f'{len(data):,}')
data = data.sort_values(by=['Implementing Jurisdictions', 'Affected Jurisdictions', 'Affected Products', 'years']).reset_index(drop=True)




# Taking care of the Green and Red interventions
data['Harmful'] = (data['GTA Evaluation'] == 'Red').astype(int)
data['Liberalizing'] = (data['GTA Evaluation'] == 'Green').astype(int)
data.drop(columns=['GTA Evaluation'], inplace=True)
data = data.groupby(['Implementing Jurisdictions', 'Affected Jurisdictions', 'Affected Products', 'years']
                    )[['Harmful', 'Liberalizing']].sum().reset_index()
data.columns = ['importer', 'exporter', 'cmd', 'yrs', 'Harmful', 'Liberalizing']
print(f'{len(data):,}')

# HS6 to HS4 code
data.loc[data.cmd!='all', 'cmd'] = data.loc[data.cmd!='all', 'cmd'].str.zfill(6).str[:4]
data = data.groupby(['importer', 'exporter', 'HS4', 'yrs']
                    )[['Harmful', 'Liberalizing']].sum().reset_index()
data.columns = ['importer', 'exporter', 'HS4', 'yrs', 'Harmful', 'Liberalizing']
print(f'{len(data):,}')



# Taking care of the 'all' exporter
all_countries = data['importer'].unique().tolist()
df = data.copy()
df['exporter'] = df['exporter'].apply(lambda x: all_countries if x == 'all' else [x])
print(f'{len(df):,}')
df = df.explode('exporter').reset_index(drop=True)
print(f'{len(df):,}')

# Taking care of the 'all' HS4
all_HS4 = pd.read_csv('../TradeHorizonScan/src/Pre-processing/data/1- CEPII_Processed_HS4_2013_2023.csv')['hsCode']
all_HS4 = all_HS4.unique().tolist()
all_HS4 = [str(code).zfill(4) for code in all_HS4]  
df['HS4'] = df['HS4'].apply(lambda x: all_HS4 if x == 'all' else [x])
print(f'{len(df):,}')
df = df.explode('HS4').reset_index(drop=True)
print(f'{len(df):,}')



df = df.groupby(['importer', 'exporter', 'HS4', 'yrs']
                    )[['Harmful', 'Liberalizing']].sum().reset_index()
print(f'{len(df):,}')


















'''
import requests
import json
import pandas as pd

#impacted product: affected_products
#implementor: importer country
#affected: exporter country
#harmful 4 anf liberlising 5: gta_evaluation
#perid: event_period

api_key = "d43f4eb7352f6d555af86333d26dd3aa2b7a721b"
# Saeed: d43f4eb7352f6d555af86333d26dd3aa2b7a721b
# Xinyue: b27cbe9a8908e84834f81e616ef7b4af87a7f705
url = "https://api.globaltradealert.org/api/v1/data/"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"APIKey {api_key}"
}




#set up payload for API request
payload = {
    "limit": 1000,  
    "offset": 200,
    "request_data": { 
        "implementer": [124], #test 5 country
        "affected": [36]  # Implementor country
        #"gta_evaluation": [4, 5],  # 4: Harmful, 5: Liberalising
        # Date
        #"implementation_period": ["2015-01-01", ""]
        #"affected_flow": [1, 2, 3] # 1: Inward, 2: Outward, 3: Outward subsidy
    }
}



R = requests.post(url, headers=headers, json=payload)
print("Status Code:", R.status_code)
print("==========")
a = [i['gta_evaluation'] for i in R.json()]
a
R.json()[0].keys()
pd.DataFrame(R.json()).to_csv("../TradeHorizonScan/src/Pre-processing/data/GTA_data.csv", index=False)
gta_evaluation, implementing_jurisdictions, affected_jurisdictions, affected_sectors, affected_products, date_implemented, date_removed

print(len(R.json()))



response = R.json()[0]
response.keys()
response['intervention_id']
response






















if response.status_code == 200:
    # Save the data to a file
    with open("gta_data.json", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Data downloaded and saved successfully.")
else:
    print(f"Failed to download data. Status code: {response.status_code}")
    print(response.text)


#setup for country year, hscode
country_codes = [784, 32, 36, 58, 56, 50, 124, 757, 756, 156, 170, 218, 251, 250, 826, 320, 344, 360,
                 699, 356, 380, 392, 410, 484, 458, 566, 528, 604, 608, 682, 702, 764, 158, 840, 841,
                 842, 868, 704]
countries_code = [32, 156, 124, 608, 360]
countries_name = ['Argantina', 'China', 'Canada', 'Philippines', 'Indonesia']

years = [str(year) + "-01-01" for year in range(2013, 2024)]


HS4_path = "../TradeHorizonScan/data/HS4CODE.csv" 
HS_codes_df = pd.read_csv(HS4_path, header=None, names=['ID'], dtype={'ID': str})
HS_codes = HS_codes_df['ID'].tolist() # Convert to string 
HS_code_digits = 4
HS_codes = [f'{int(code):0{HS_code_digits}d}' for code in HS_codes] # ensure all codes are 4 digits
print("Extract attached HS4 codes:", HS_codes)






#transfer json to pandas dataframe
content_json = json.loads(response.text)
content_json.keys()

'''