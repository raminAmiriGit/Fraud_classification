import os
import requests
import numpy as np
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

#read from env variable DATABRICKS_TOKE
token = os.environ.get("DATABRICKS_TOKEN")
def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://fevm-ramin-serverless-aws.cloud.databricks.com/serving-endpoints/mistplay-fraud-features/invocations'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    # ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    # data_json = json.dumps(ds_dict, allow_nan=True)
    # response = requests.request(method='POST', headers=headers, url=url, data=dataset)
    response = requests.post(url, headers=headers, json=dataset, timeout=30)

    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

def score_model_concurrent(dataset, concurrency: int):
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(score_model, dataset) for _ in range(concurrency)]
        for future in as_completed(futures):
            results.append(future.result())
    return results

payload ={'dataframe_records': [{'account_id': '3750', 'device_id': '1088'},
  {'account_id': '3751', 'device_id': '1433'},
  {'account_id': '3752', 'device_id': '2888'},
  {'account_id': '3753', 'device_id': '2739'},
  {'account_id': '3754', 'device_id': '1475'}]}

result = score_model(payload)
print(result)


###########################################################
###########################################################
# Concurrent invocation (keeps the single-call above)
concurrent_calls = 20  # set your desired concurrency



concurrent_results = score_model_concurrent(payload, concurrent_calls)
print(f"Concurrent results: {concurrent_results} responses")