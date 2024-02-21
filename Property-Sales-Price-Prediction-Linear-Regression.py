import os
import pandas as pd
import numpy as np
import seaborn as sns

# Increase the print output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set working directory
os.chdir("C:/Users/dipti/Documents/IMR/Data_Linear_Regression/")

# Read in the data
rawDf = pd.read_csv("PropertyPrice_Data.csv")
predictionDf = pd.read_csv("PropertyPrice_Prediction.csv")


# Before we do anything, we need to divide our Raw Data into train and test sets (validation)
from sklearn.model_selection import train_test_split
trainDf, testDf = train_test_split(rawDf, train_size=0.8, random_state = 150)

# Create Source Column in both Train and Test
trainDf['Source'] = "Train"
testDf['Source'] = "Test"
predictionDf['Source'] = "Prediction"

# Combine Train, Test and Prediction 
fullRaw = pd.concat([trainDf, testDf, predictionDf], axis = 0)
fullRaw.shape


# Lets drop "Id" column from the data as it is not going to assist us in our model
fullRaw = fullRaw.drop(['Id'], axis = 1) 
# fullRaw.drop(['Id'], axis = 1, inplace = True) 

# Check for NAs
fullRaw.isnull().sum()

# Check data types of the variables
fullRaw.dtypes



############################
# Univariate Analysis: Missing value imputation
############################

# Garage variable (Categorical)
tempMode = fullRaw.loc[fullRaw["Source"] == "Train", "Garage"].mode()[0] 
# tempMode = trainDf["Garage"].mode()[0]

fullRaw["Garage"].fillna(tempMode, inplace = True) 


# Garage_Built_Year (Continuous)

tempMedian = trainDf["Garage_Built_Year"].median() # Same as above
tempMedian
fullRaw["Garage_Built_Year"] = fullRaw["Garage_Built_Year"].fillna(tempMedian)   



# All NAs should be gone now
fullRaw.isnull().sum()


############################
# Bivariate Analysis Continuous Variables: Scatterplot
############################

corrDf = fullRaw[fullRaw["Source"] == "Train"].corr()
# corrDf.head()
sns.heatmap(corrDf, 
        xticklabels=corrDf.columns,
        yticklabels=corrDf.columns, cmap='RdBu')



############################
# Bivariate Analysis Categorical Variables: Boxplot
############################


# First option: Do it manually for each variable
sns.boxplot(y = trainDf["Sale_Price"], x = trainDf["Road_Type"]) # Plot for Road_Type
sns.boxplot(y = trainDf["Sale_Price"], x = trainDf["Property_Shape"]) # Plot for Property_Shape

categoricalVars = trainDf.columns[trainDf.dtypes == object]
categoricalVars

# Second option: Run a for loop and create multiple plots
from matplotlib.pyplot import figure
for colName in categoricalVars:
    figure()    
    sns.boxplot(y = trainDf["Sale_Price"], x = trainDf[colName])

# Third option: Run a for loop to dump all the plots in a pdf file
from matplotlib.backends.backend_pdf import PdfPages
fileName = "C:/Users/dipti/Documents/IMR/Data_Linear_Regression/Categorical_Variables_Analysis.pdf"
pdf = PdfPages(fileName)
for colNumber, colName in enumerate(categoricalVars): # enumerate gives key, value pair
    # print(colNumber, colName)
    figure()
    sns.boxplot(y = trainDf["Sale_Price"], x = trainDf[colName])
    pdf.savefig(colNumber+1) # colNumber+1 is done to ensure page numbering starts from 1 (and NOT 0)
    
pdf.close()

    

############################
# Dummy variable creation
############################

fullRaw2 = pd.get_dummies(fullRaw, drop_first = True)  # drop_first = True will ensure you get n-1 dummies for each categorcial var
# 'Source'  column will change to 'Source_Train' & 'Source_Test' and will contain 0s and 1s

fullRaw2.shape
fullRaw.shape

############################
# Add Intercept Column
############################


from statsmodels.api import add_constant
fullRaw2 = add_constant(fullRaw2)
fullRaw2.shape

############################
# Sampling
############################

# Step 1: Divide into Train, Test and Prediction Dfs
trainDf = fullRaw2[fullRaw2['Source_Train'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()

# trainDf = fullRaw2[fullRaw2['Source_Train'] == 1]
# trainDf.drop(['Source_Train', 'Source_Test'], axis = 1, inplace = True)

testDf = fullRaw2[fullRaw2['Source_Test'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()
predictionDf = fullRaw2[(fullRaw2['Source_Train'] == 0) & 
                        (fullRaw2['Source_Test'] == 0)].drop(['Source_Train', 'Source_Test'], axis = 1).copy()

trainDf.shape
testDf.shape
predictionDf.shape

# Step 2: Divide into Xs (Indepenedents) and Y (Dependent)
trainX = trainDf.drop(['Sale_Price'], axis = 1).copy()
trainY = trainDf['Sale_Price'].copy()
testX = testDf.drop(['Sale_Price'], axis = 1).copy()
testY = testDf['Sale_Price'].copy()
predictionDf = predictionDf.drop(["Sale_Price"], axis = 1)

trainX.shape
trainY.shape
testX.shape
testY.shape


#########################
# VIF check
#########################

from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF = 5 
maxVIFCutoff = 5 
counter = 1
highVIFColumnNames = []

while (tempMaxVIF >= maxVIFCutoff):

    # Create an empty temporary df to store VIF values
    tempVIFDf = pd.DataFrame()
    
    # Calculate VIF using list comprehension
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
    
    # Create a new column "Column_Name" to store the col names against the VIF values from list comprehension
    tempVIFDf['Column_Name'] = trainXCopy.columns
    
    # Drop NA rows from the df - If there is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    # Sort the df based on VIF values, then pick the top most column name (which has the highest VIF)
    tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,1]
    
    
    # Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,0]
    
    
    print(tempColumnName)
    
    if (tempMaxVIF >= maxVIFCutoff): 
        
        
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highVIFColumnNames.append(tempColumnName)
    
    counter = counter + 1

highVIFColumnNames


highVIFColumnNames.remove('const') 
trainX = trainX.drop(highVIFColumnNames, axis = 1)
testX = testX.drop(highVIFColumnNames, axis = 1)
predictionDf = predictionDf.drop(highVIFColumnNames, axis = 1)

trainX.shape
testX.shape
predictionDf.shape

#########################
# Model Building
#########################

from statsmodels.api import OLS
m1ModelDef = OLS(trainY, trainX) 
m1ModelBuild = m1ModelDef.fit() 
m1ModelBuild.summary() 


# Extract/ Identify p-values from model
dir(m1ModelBuild)
m1ModelBuild.pvalues

#########################
# Model Optimization
#########################


tempMaxPValue = 0.05
maxPValueCutoff = 0.05
trainXCopy = trainX.copy()
counter = 1
highPValueColumnNames = []


while (tempMaxPValue >= maxPValueCutoff):
    
    print(counter)    
    
    tempModelDf = pd.DataFrame()    
    Model = OLS(trainY, trainXCopy).fit()
    tempModelDf['PValue'] = Model.pvalues
    tempModelDf['Column_Name'] = trainXCopy.columns
    tempModelDf.dropna(inplace=True) 
    tempColumnName = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,1]
    tempMaxPValue = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,0]
    
    if (tempMaxPValue >= maxPValueCutoff): 
        print(tempColumnName, tempMaxPValue)    
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highPValueColumnNames.append(tempColumnName)
    
    counter = counter + 1

highPValueColumnNames

# Check final model summary
Model.summary()
trainX = trainX.drop(highPValueColumnNames, axis = 1)
testX = testX.drop(highPValueColumnNames, axis = 1)
predictionDf = predictionDf.drop(highPValueColumnNames, axis = 1)

trainX.shape
testX.shape

# Build model on trainX, trainY (after removing insignificant columns)
Model = OLS(trainY, trainX).fit()
Model.summary()


#########################
# Model Prediction
#########################


Test_Pred = Model.predict(testX)
Test_Pred[0:6]
testY[:6]

#########################
# Model Diagnostic Plots (Validating the assumptions)
#########################

import seaborn as sns

# Homoskedasticity check
sns.scatterplot(Model.fittedvalues, Model.resid) 

# Normality of errors check
sns.distplot(Model.resid) 

#########################
# Model Evaluation
#########################

# RMSE
np.sqrt(np.mean((testY - Test_Pred)**2))


# MAPE (Mean Absolute Percentage Error)
(np.mean(np.abs(((testY - Test_Pred)/testY))))*100

#########################
# Model Prediction (of prediction csv file)
#########################

predictionDf["Predicted_Sale_Price"] = Model.predict(predictionDf.drop(["Sale_Price"], axis = 1))

predictionDf.to_csv("predictionDf.csv")



