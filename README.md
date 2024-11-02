# Social Value Analysis Using Sky Data
## Constructing_network.py 
- Make sure you store all your data files in one folder. 
- This function assumes data files as csv, but you can change the file i/o lines if you are using other types of data files
- The write function exports network data into csv. This is for in case if you need to use gephi for network visualization. 
- We will add an pickle export funciton if needed. 

## calculate_SV.py
Here are the input file instructions:
#### ValueFeaturesFile
This file should be a .csv file that stores user id, nonsocial features, and social features

#### NetworkFile
This file should be a .csv file that stores source nodes, target nodes, and edge-weights. Edgeweights are optional. 

Here are the imput parameter definitons
#### EmptyNeighborhoodFeatureValues - 
List of Values that specifies a default value to be used when a particular social feature is missing or when there is an empty 
neighborhood (e.g., no connections or interactions for a user).

#### OneHopNetworkNeighborFeatures - 
List of Indices of columns from ValueFeaturesFile that specifies features that are social in nature

#### idColumn - 
The index of columns in ValueFeaturesFile that stores user ID (random_id in the case of sky)

#### ValueColumn - 
The index of the user value response column in ValueFeaturesFile (If no value provided then defaults to the last column in ValueFeaturesFile)
