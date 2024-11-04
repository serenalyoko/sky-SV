# Import libraries
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

def query_sky_data(path_to_credential, query_string, out_file)
    # Get API key from  "APIs & Services" Panel
    # https://console.cloud.google.com/apis/credentials?authuser=1&project=skydev-5402c&folder=&organizationId=
    credentials = service_account.Credentials.from_service_account_file(
    path_to_credential)

    # Connect to db
    try:
        client = bigquery.Client(
            credentials=credentials,
            project=credentials.project_id,
        )
        print('Successfully connect to BigQuery!')
    except Exception:
        print('Failed to connect BigQuery.' + ['error:big_query_conn'])

    # Run SQL Query
    query_job = client.query(query_string)
    res = query_job.result()
    print(res)
    # Create a dataframe from the query result
    df = res.to_dataframe()
    print(df.columns)

    # Save the dataframe to a CSV file
    df.to_csv(out_file, index=False)

if __name__ == "__main__":
    path_to_credential = "/Users/siyizhou/Documents/2024Fall/pythonProject/api_credential.json"
    query_string = (
        'SELECT random_id1, random_id0, date_pst, connect_via,playtime_1on1 FROM `sky-usc-analytics.sky_usc_exports.accountlive_fact_relationship` WHERE date_pst = "2022-02-01"')
    out_file = "query_result.csv"
    query_sky_data(path_to_credential, query_string,out_file )
