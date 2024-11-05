import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import math
import shap


"""
This Document sets up some parameters and configurations for handling social features in your dataset, 
along with a default value to use for empty neighborhoods.

Variable definitions:

EmptyNeighborhoodFeatureValues - 
List of Values that specifies a default value to be used when a particular social feature is missing or when there is an empty 
neighborhood (e.g., no connections or interactions for a user).

OneHopNetworkNeighborFeatures - 
List of Indices of columns from ValueFeaturesFile that specifies features that are social in nature

idColumn - 
The index of columns in ValueFeaturesFile that stores user ID (random_id in the case of sky)

ValueColumn - 
The index of the user value response column in ValueFeaturesFile (If no value provided then defaults to the last column in ValueFeaturesFile)

"""
def computeSV(ValueFeaturesFile, NetworkFile,  OneHopNetworkNeighborFeatures,
                      EmptyNeighborhoodFeatureValues, idColumn = 0, ValueColumn = None,
              ResultsFileName="SVResults.csv" ):
    """
    This code block does further data preprocessing and the training of a Random Forest regression model using features 
    and a target variable.
    """
    svdata = pd.read_csv(ValueFeaturesFile)
    ndata = pd.read_csv(NetworkFile)

    # Data prep
    # Remove records with missing values
    svdata = svdata.dropna()
    #svdata = svdata.drop_duplicates(subset=['DONOR_ID'])
    ndata = ndata.dropna()

    #If ValueColumn is None, it sets targetVariable to the last column index in svdata using len(svdata.columns) - 1.
    if ValueColumn == None :
        targetVariable  = len(svdata.columns)-1 #default to last column
    else:
        targetVariable = ValueColumn

    #Feature and Target Selection. Prepare X and Y for model
    a = set(range(len(svdata.columns))) #Creates a set a containing the indices of all columns in svdata
    b = {idColumn,targetVariable} #Creates a set b containing the indices of the idColumn and targetVariable columns. These two columns should be excluded from the features used in training.
    c= list(a-b)#Computes the set difference a - b, resulting in the indices of columns that are neither idColumn nor targetVariable. Converts this set to a list c.

    svdatax = svdata.iloc[:, c] #Contains the independent variables (features) to be used in the model
    svdatay = svdata.iloc[:, targetVariable] #Contains the dependent variable (target) to be predicted


    # Build random forest model over whole dataset
    #--------------------without tuning
    nTree = 100 #Sets nTree to 100, indicating the number of decision trees in the Random Forest model.
    sampFrac = 0.25 #Sets sampFrac to 0.25, indicating that 25% of the data will be used for training each tree.
    numsamples = math.ceil(sampFrac*len(svdatax.index)) #Calculates the number of samples (numsamples) as 25% of the total rows in svdatax. Uses math.ceil() to ensure this value is rounded up to the nearest integer.

    print("Building forest...")
    rf = RandomForestRegressor(n_estimators = nTree)

    # Train the model on training data
    rf.fit(svdatax, svdatay) #Trains the Random Forest model on the features (svdatax) and target (svdatay).
    y_pred_rf = rf.predict(svdatax) #Uses the trained model to predict the values of svdatay based on the features in svdatax.

    print("done.")

    #I built the tree again for SHAP visualization to understand the impact of features on Y.
    # This part if optional. Not for SV calculation

    X_train, X_test, y_train, y_test = train_test_split(svdatax, svdatay, test_size=0.20, random_state=3)

    """
    train_test_split is a function from the sklearn.model_selection module (part of Scikit-Learn), a popular machine 
    learning library in Python.
    
    In this case, test_size=0.25 means 25% of the total data will be used for testing, and the remaining 80% will be 
    used for training.
    
    random_state=3. Sets the random seed for shuffling the data before splitting. Setting random_state ensures that 
    the train-test split is reproducible. Every time the code is run with the same random_state, the data will be split 
    in the same way.
    
    """
    r2_score = sklearn.metrics.r2_score(svdatay, y_pred_rf)
    #r2_score measures the proportion of variance in the dependent variable that is predictable from the independent variables.

    #This code block creates a copy of the target variable (svdatay) to manipulate it without affecting the original data.
    y_actual = svdatay.copy()
    y_actual = y_actual.replace(to_replace = 0, value=0.01)  #replace 0s by 0.01

    """"
    Here, any values in y_actual that are equal to 0 are replaced by 0.01. Why? Taking the logarithm of zero is 
    mathematically undefined, so this step avoids that error by substituting small positive values (0.01) instead of zeros.
    """

    logged_y_actual= list(np.log10(y_actual)) #Apply a logarithmic transformation to the modified target values (y_actual) using np.log10 (base-10 logarithm).

    h=np.histogram(logged_y_actual) #do binning. Binning allows you to group continuous values into categories (bins), making it easier to evaluate model performance in different ranges of the data.
    breaks = h[1] #Extract the breakpoints (bin edges) from the histogram.
    bin_no = [-1]*len(logged_y_actual) #initialize Bin Numbers for Each Data Point:

    """
    The main purpose of this section is to assign a weight to each value in the list logged_y_actual based on which bin 
    the value falls into.
    The code achieves this by using histogram binning defined by the breaks list.
    """

    #This loop iterates through each value (num) in logged_y_actual by its index j.
    for j in range(0,len(logged_y_actual)):
        num = logged_y_actual[j]
        flag = False

    #This loop iterates through each bin defined by the bin edges breaks
        for i in range(0, len(breaks)-1): # check which bin it falls into
            left = breaks[i]
            right = breaks[i+1]
            if (left<=num) & (num<right): # matches
                bin_no[j] = 10**i # this is weight, bin no is i.
                flag=True
            break

    #This condition handles the edge case where a value num is exactly equal to the rightmost edge of the last bin (breaks[len(breaks)-1]).
        if flag==False and num==breaks[len(breaks)-1]:
            bin_no[j] = 10**(len(breaks)-1)

    #The use of 10**i as a weight provides a numeric value that increases exponentially with the bin index. It allows the bins to be distinguished by magnitude rather than a simple index.
    smoothing_factor = 0.01 #This small value is used to prevent division by zero when calculating the relative error. It stabilizes the calculation when the denominator is close to zero.
    relative_error = [0]*len(y_actual) #Initializes a list called relative_error with the same length as y_actual, filled with zeros. This list will be used to store the relative error for each data point
    #Convert y_actual and y_pred_rf to Lists. y_actual and y_pred_rf are converted to lists to ensure that operations like indexing and iteration work correctly.
    y_actual = list(y_actual)
    y_pred_rf = list(y_pred_rf)

    #Loop Through Each Data Point to Calculate Relative Error
    for i in range(0,len(y_actual)):
        abs_error = abs(y_actual[i]-y_pred_rf[i])
        rel_err = abs_error / (y_actual[i]+smoothing_factor)
        relative_error[i] = rel_err * bin_no[i]


    #Weighted Relative Error
    weighted_err = sum(relative_error)/sum(bin_no)
    accuracy_perc = 100-weighted_err*100


    # Prepare data with no neighbors being simulated
    nonNeighborData = svdata
    #Creates a copy of the original svdata to work on. This new dataframe (nonNeighborData) will be used to simulate a scenario where social features are set to default values (no neighbor influence).

    numSocialFeatures = len(OneHopNetworkNeighborFeatures)
    #Calculates the number of social features (columns) specified in socialFeatures.

    defaultSocialDataBlock = None
    #This is currently set to None but could be used later to store a default block of social data.

    numRecords = len(svdata.index)
    #Stores the number of records (rows) in the dataset svdata

    # Calculate Weighted Error and Accuracy
    #Set Default Values for Social Features (No Neighbor Influence)
    for j in range(0,numSocialFeatures) :
        feature = OneHopNetworkNeighborFeatures[j]
        nonNeighborData.iloc[:, feature] = EmptyNeighborhoodFeatureValues[j]


    # Estimate target variable when simulating no neighbors
    #This line uses the random forest model (rf) to predict target variable values for the dataset nonNeighborData, which simulates a scenario where social features are set to default values (indicating no social influence).
    estimatedNoSocialY = rf.predict(nonNeighborData.iloc[:, c])

    # Estimate network power as difference in actual and no social simulated target variable values.
    #This block computes the difference between the actual target variable values in svdata and the predicted values (estimatedNoSocialY) from the no-neighbor simulation.
    print("\nComputing sv ...")
    temp = svdata.iloc[:,targetVariable] - estimatedNoSocialY

    """
    This section prepares a DataFrame (networkPowerFull) that combines the ID column and the computed network power values.
    temp.mask(temp < 0, 0): This modifies the temp array by masking (i.e., replacing) any negative values with 0.
    This means that if the network power (the difference calculated) is negative, it is set to 0 to indicate that 
    no network power exists in such cases.
    If you reset this value, you allow negative SV to exist. By modifying this part of the code, you can have negative SV
    
    """
    data = [svdata.iloc[:,idColumn], temp.mask(temp<0, 0)]

    #Concatenates the list of Series along the columns (axis=1) to create a new DataFrame called networkPowerFull.
    #networkPowerFull essentially captures the total influence of the presence of network on people's behaviors.
    networkPowerFull = pd.concat(data, axis=1)

    networkPowerFull.columns = ['dest', 'NetworkPower_dest']

    """
    This line renames the columns of the ndata DataFrame to more meaningful names
    'src': Represents the source node in the network.
    'dest': Represents the destination node in the network.
    'weight': Represents the weight or strength of the connection from the source to the destination.
    """
    ndata.columns = ['src', 'dest', 'weight']

    grouped = ndata.groupby('dest') # groups the ndata DataFrame by the destination nodes
    grp_sum = grouped.aggregate(np.sum) #Sums the weights for each destination node, effectively calculating the total incoming weight for each destination

    #This resets the index of the grp_sum DataFrame, converting the grouped index back into a regular column. This is necessary for future merging operations.
    grp_sum.reset_index(inplace=True)

    #If the src column doesn't exist, the except block prevents the code from throwing an error. This ensures that the code continues to run smoothly.
    try:
        grp_sum = grp_sum.drop(['src'], axis=1) #deleting column sourceid
    except:
        pass

    #merges the original ndata DataFrame with the aggregated grp_sum DataFrame
    nndata = ndata.merge(grp_sum, how='left', on='dest', suffixes = ['_pair','_sumForDest'])

    #calculates the normalized edge weight for each row in nndata.
    #The formula indicates how significant each edge is compared to the total weight directed toward the destination.
    nndata['normalized_ew'] = nndata['weight_pair'] / nndata['weight_sumForDest']

    #fills any NaN values in nndata with 0.
    #This is particularly important if there were any destinations that had no incoming connections, resulting in NaN when calculating normalized_ew
    nndata=nndata.fillna(0)

    #merges nndata with the networkPowerFull DataFrame (which contains network power data calculated previously)
    #ewsvresult stands for error weighted SV results. Note how it is merged on destinations
    ewsvresult = nndata.merge(networkPowerFull, how='left', on='dest')

    # This line calculates a new column called ewsv that represents the Edge Weight and Network Power Value for each destination.
    ewsvresult['ewsv'] = ewsvresult['normalized_ew']*ewsvresult['NetworkPower_dest']

    #This segment creates a DataFrame (tempdf) to hold the source node IDs, destination node IDs, and their corresponding social value (ewsv).
    tempdf = pd.DataFrame()
    tempdf['source'] = ewsvresult['src']
    tempdf['dest'] = ewsvresult['dest']
    tempdf['sv'] = ewsvresult["ewsv"]
    tempdf.to_csv(ResultsFileName[:-4] + "_directed_sv.csv", header=True, index=None)


    #compute SV
    #This block computes the Social Value by grouping the ewsvresult DataFrame by the source node (src).
    grouped = ewsvresult.groupby('src') #groupby source node
    grp_sum = grouped.aggregate(np.sum) #After grouping, it sums the ewsv values to get the total social value for each source node.
    grp_sum.reset_index(inplace=True)

    svresult = pd.DataFrame()
    svresult["userID"]=grp_sum['src']
    svresult["SocialValue"] = grp_sum['ewsv']


    #compute Influencability
    #Similar to the social value calculation, this block computes the Influenceability by grouping the ewsvresult DataFrame by the destination node (dest).
    grouped = ewsvresult.groupby('dest') #groupby dest
    grp_sum = grouped.aggregate(np.sum)
    grp_sum.reset_index(inplace=True)

    infresult = pd.DataFrame()
    infresult["userID"]=grp_sum['dest']
    infresult["Influenceability"] = grp_sum['ewsv']

    #This segment creates a DataFrame (psresult) containing userID and their corresponding Personal Spend values from the original svdata DataFrame.
    psresult= pd.DataFrame()
    psresult["userID"]=svdata.iloc[:,idColumn]
    psresult["PersonalSpend"] = svdata.iloc[:,targetVariable]

    #This merges the svresult (Social Value) with psresult (Personal Spend) on the userID.
    merge1 = svresult.merge(psresult, on="userID", how="outer")
    merge1 = merge1.fillna(0)


    resultstable = merge1.merge(infresult, on="userID", how="outer")
    resultstable = resultstable.fillna(0)

    #Asocial Value: The difference between PersonalSpend and Influenceability.
    resultstable['AsocialValue'] = resultstable['PersonalSpend']-resultstable['Influenceability']

    #Network Power: The sum of AsocialValue and SocialValue.
    resultstable['NetworkPower'] = resultstable['AsocialValue'] + resultstable['SocialValue']

    #Total Value: The sum of SocialValue, AsocialValue, and Influenceability.
    resultstable['TotalValue'] = resultstable['SocialValue'] + resultstable['AsocialValue'] + resultstable['Influenceability']

    # generate stats for each column.Minimum, Maximum, Standard Deviation, Mean, Total for the different metrics.
    metric_stat = ["Social Value", "Asocial Value", "Influenceability", "Network Power", "Personal Spend", "Total Value"]

    minimum = [min(resultstable['SocialValue']), min(resultstable['AsocialValue']), min(resultstable['Influenceability']), min(resultstable['NetworkPower']), min(resultstable['PersonalSpend']), min(resultstable['TotalValue'])]

    maximum= [max(resultstable['SocialValue']), max(resultstable['AsocialValue']), max(resultstable['Influenceability']), max(resultstable['NetworkPower']), max(resultstable['PersonalSpend']), max(resultstable['TotalValue'])]

    std = [np.std(resultstable['SocialValue']), np.std(resultstable['AsocialValue']), np.std(resultstable['Influenceability']), np.std(resultstable['NetworkPower']), np.std(resultstable['PersonalSpend']), np.std(resultstable['TotalValue'])]

    mean= [np.mean(resultstable['SocialValue']), np.mean(resultstable['AsocialValue']), np.mean(resultstable['Influenceability']), np.mean(resultstable['NetworkPower']), np.mean(resultstable['PersonalSpend']), np.mean(resultstable['TotalValue'])]

    total = [sum(resultstable['SocialValue']), sum(resultstable['AsocialValue']), sum(resultstable['Influenceability']), sum(resultstable['NetworkPower']), sum(resultstable['PersonalSpend']), sum(resultstable['TotalValue'])]

    data = {'Metric':metric_stat, 'Min':minimum, 'Max':maximum, 'Std': std, 'Mean': mean, 'Total':total}
    stat_df = pd.DataFrame(data)

    #This calculates the percentage of total social value relative to the sum of social and asocial values, giving insight into the proportion of social influence compared to personal spending.
    social_percentage = sum(resultstable['SocialValue']) / (sum(resultstable['SocialValue']) + sum(resultstable['AsocialValue']))

    print("done, writing results to file\n")

    resultstable.to_csv(ResultsFileName, header=True, index=None)

    ret = (stat_df, social_percentage*100, r2_score, accuracy_perc)
    """
    social_percentage * 100: The percentage of social value expressed as a percentage.
    r2_score: A measure of how well the model predicts outcomes.
    accuracy_perc: The accuracy percentage calculated earlier in the code.
    """

    print((social_percentage*100, r2_score, accuracy_perc)) ## Algorithm vs network driven.
    print(stat_df)
    print(ret)
    #The main purpose of this line is to visualize the distribution of the values in y_actual.
    #By plotting the histogram, you can quickly see how the data is spread out, identify any peaks, gaps, or skewness in the distribution, and understand the underlying characteristics of the data.

    plt.hist(y_actual, bins=40)
    """
    y_actual: the actual target variable values from your model (after any necessary preprocessing, such as replacing 0s with 0.01).
    bins=40: This parameter specifies the number of bins to divide the data into. Bins are the intervals that group the data points.
    By setting bins=40, you are instructing the histogram to create 40 equally spaced intervals across the range of y_actual.
    Each bin will then count how many data points fall within that interval.
    """
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test,alpha = 0.5)
