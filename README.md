# Social Value Analysis Using Sky Data

The `data_prep.py` contains two steps to prepare your SV calculation: 1) contructing network file and a unique user_id file called node list, 2) generate social features file

## Constructing Network
- Make sure you store all your data files in one folder. 
- This function assumes data files as csv, but you can change the file i/o lines if you are using other types of data files
- The write function exports network data into csv. This is for in case if you need to use gephi for network visualization. 
- We will add an pickle export funciton if needed.

## Match user features with users in the network
### Match user after download data from BigQuery
- First query columns "random_id" and any features you would like to use for SV calculation
- Save your results to a .csv file
- Execute the `get_social_features()` function
  
### Querying social features with matched user ID in BigQuery
You need to first upload your own unique node list file to sky_usc_datalab
Then excute the following
``` 
SELECT *
FROM `sky_usc_exports.user_profile` AS main_table
JOIN `sky_usc_datalab.your_own_node_list` AS node_list
ON main_table.random_id = node_list.node 
WHERE first_level_loaded_date_pst >= "your-start-time" AND first_level_loaded_date_pst < "your-end-time"
``` 

## calculate_SV.py
### Here are the input file instructions:
#### ValueFeaturesFile
This file should be a .csv file that stores user id, nonsocial features, and social features

#### NetworkFile
This file should be a .csv file that stores source nodes, target nodes, and edge-weights. Edgeweights are optional. 

### Here are the imput parameter definitons
#### EmptyNeighborhoodFeatureValues - 
List of Values that specifies a default value to be used when a particular social feature is missing or when there is an empty 
neighborhood (e.g., no connections or interactions for a user).

#### OneHopNetworkNeighborFeatures - 
List of Indices of columns from ValueFeaturesFile that specifies features that are social in nature

#### idColumn - 
The index of columns in ValueFeaturesFile that stores user ID (random_id in the case of sky)

#### ValueColumn - 
The index of the user value response column in ValueFeaturesFile (If no value provided then defaults to the last column in ValueFeaturesFile)
