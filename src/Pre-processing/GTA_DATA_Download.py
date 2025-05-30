import requests
import json
import pandas as pd

#impacted product: affected_products
#implementor: importer country
#affected: exporter country
#harmful 4 anf liberlising 5: gta_evaluation
#perid: event_period

api_key = "b27cbe9a8908e84834f81e616ef7b4af87a7f705"
url = "https://api.globaltradealert.org/api/v1/data/"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"APIKey {api_key}"
}




#set up payload for API request
payload = {
    "limit": 1000,  
    "offset": 0,
    "request_data": { 
        "implementer": country_codes, #test 5 country
        "affected": [842, 124],  # Implementor country
        "gta_evaluation": [4, 5],  # 4: Harmful, 5: Liberalising
        #"affected_products": [2790],
        "affected_flow": [1, 2, 3] # 1: Inward, 2: Outward, 3: Outward subsidy
    }
}



response = requests.post(url, headers=headers, json=payload)
print("Status Code:", response.status_code)
response = response.json()[0]
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