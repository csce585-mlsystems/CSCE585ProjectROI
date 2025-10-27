# Purpose: This file will contain code that helps with <insert subsystem here>

# Important things that these files will do: a) Obtain the dataset, b) Convert it into DataFrame, c) At least transform the data to produce a modified version of dataset that contains the attributes we need, d) (cont here if applicable)

# Body of neccessary imports
import numpy as np
import pandas as pd

# end of body of neccessary imports

# Body of code that brute forced data processing

# a) Deriving filepaths to transform .csv files into DataFrames:
print("Deriving filepaths to transform .csv files into DataFrame")
dataset1 = pd.read_csv('DataPreparation/cleanedDataUnmodified/apple_Metrics.csv', index_col='Metric')
dataset2 = pd.read_csv('DataPreparation/cleanedDataUnmodified/google_Metrics.csv', index_col='Metric')
dataset3 = pd.read_csv('DataPreparation/cleanedDataUnmodified/amazon_Metrics.csv', index_col='Metric')

#dataset1per = pd.read_table('DataPreparation/personalAppleVersion.csv') 
#dataset2per = pd.read_table('DataPreparation/personalGoogleVersion.csv') 
#dataset3per = pd.read_table('DataPreparation/personalAmazonVersion.csv')


# end of a)

# b) Creating a copy of the dataframe(s) and then adding the neccessary columns to the copied version

# Formulas being computed: a) P/E = \frac{Price}{Diluted EPS}, b) P/B = \frac{Price}{\frac{Total Stockholder Equity}{Outstanding Shares}}, c) D/E = \frac{Total Debt}{Total Stockholder Equity}, and d) FCF = Operating Cash Flow + Captial Expenditure 
#0                                    Metric
#1                       Operating Cash Flow
#2                       Capital Expenditure
#3                            Free Cash Flow
#4                                Total Debt
#5                 Common Stockholder Equity
#6   Total Liabilities Net Minority Interest
#7                              Total Assets
#8                        Shares Outstanding
#9            Net Income Common Stockholders
#10                   Diluted Average Shares
#11                              Diluted EPS
#12                                    Price
dataset1cpy = dataset1.T
dataset2cpy = dataset2.T
dataset3cpy = dataset3.T
dataset1cpy['P/E'] = dataset1cpy['Price']/dataset1cpy['Diluted EPS']# insert formula here using columns == dataset1cpy[13] = dataset1cpy[12]/dataset1cpy[11]
dataset1cpy['P/B'] = dataset1cpy['Price'] * (dataset1cpy['Shares Outstanding']/dataset1cpy['Common Stockholder Equity'])# insert formula here using columnsdataset1cpy['D/E'] = # insert formula here using columns == dataset1cpy[14] = dataset1cpy[12]* (dataset1cpy[9]/dataset1cpy[11])
dataset1cpy['D/E'] = dataset1cpy['Total Debt']/dataset1cpy['Common Stockholder Equity']# insert formula here using columns == dataset1cpy[15] = dataset1cpy[4]/dataset1cpy[5]
dataset1cpy['FCF'] = dataset1cpy["Operating Cash Flow"] + dataset1cpy["Capital Expenditure"] # insert formula here using columns == dataset1cpy[16] = dataset1cpy[1] + dataset1cpy[2]  
dataset2cpy['P/E'] = dataset2cpy['Price']/dataset2cpy['Diluted EPS']# insert formula here using columns
dataset2cpy['P/B'] = dataset2cpy['Price'] * (dataset2cpy['Shares Outstanding']/dataset2cpy['Common Stockholder Equity'])# insert formula here using columnsdataset1cpy['D/E'] = # insert formula here using columns
dataset2cpy['D/E'] = dataset2cpy['Total Debt']/dataset2cpy['Common Stockholder Equity']# insert formula here using columns
dataset2cpy['FCF'] = dataset2cpy["Operating Cash Flow"] + dataset2cpy["Capital Expenditure"] # insert formula here using columns
dataset3cpy['P/E'] = dataset3cpy['Price']/dataset3cpy['Diluted EPS']# insert formula here using columns
dataset3cpy['P/B'] = dataset3cpy['Price'] * (dataset3cpy['Shares Outstanding']/dataset3cpy['Common Stockholder Equity'])# insert formula here using columnsdataset1cpy['D/E'] = # insert formula here using columns
dataset3cpy['D/E'] = dataset3cpy['Total Debt']/dataset3cpy['Common Stockholder Equity']# insert formula here using columns
dataset3cpy['FCF'] = dataset3cpy["Operating Cash Flow"] + dataset3cpy["Capital Expenditure"] # insert formula here using columns

# Dropping the extraneous columns: 
dataset1cpy.drop(['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow','Total Debt', 'Common Stockholder Equity', 'Total Liabilities Net Minority Interest', 'Total Assets','Shares Outstanding', 'Net Income Common Stockholders','Diluted Average Shares'],axis=1)
dataset2cpy.drop(['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow','Total Debt', 'Common Stockholder Equity', 'Total Liabilities Net Minority Interest', 'Total Assets','Shares Outstanding', 'Net Income Common Stockholders','Diluted Average Shares'],axis=1)
dataset3cpy.drop(['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow','Total Debt', 'Common Stockholder Equity', 'Total Liabilities Net Minority Interest', 'Total Assets','Shares Outstanding', 'Net Income Common Stockholders','Diluted Average Shares', 'Diluted EPS', 'Price'],axis=1)

# end of Dropping the extraneous columns

# Creating a dataframe that references company names with their associated metrics: 
companyDataset = DataFrame(np.zeros((3,4)),index=["Apple", "Amazon", "Google"], columns=['P/E', 'P/B', 'D/E', 'FCF'])

# Computing mean for each column to be loaded into companyDataset: 

companyDataset['P/E'].iloc[0] = dataset1cpy['P/E'].mean()
companyDataset['P/B'].iloc[0] = dataset1cpy['P/B'].mean()
companyDataset['D/E'].iloc[0] = dataset1cpy['D/E'].mean()
companyDataset['FCF'].iloc[0] = dataset1cpy['FCF'].mean()
companyDataset['P/E'].iloc[1] = dataset2cpy['P/E'].mean()
companyDataset['P/B'].iloc[1] = dataset2cpy['P/B'].mean()
companyDataset['D/E'].iloc[1] = dataset2cpy['D/E'].mean()
companyDataset['FCF'].iloc[1] = dataset2cpy['FCF'].mean()
companyDataset['P/E'].iloc[2] = dataset3cpy['P/E'].mean()
companyDataset['P/B'].iloc[2] = dataset3cpy['P/B'].mean()
companyDataset['D/E'].iloc[2] = dataset3cpy['D/E'].mean()
companyDataset['FCF'].iloc[2] = dataset3cpy['FCF'].mean()
# end of computing mean for each column to be loaded into companyDataset

# Getting ready to create training data for ML Model
companyDatasetLabeledVersion = companyDataset.copy()
companyDatasetLabeledVersion['Recommended'] = (True,False,True)

companyDatasetLabeledVersion.to_csv('DataPreparation/firstVersionTrainingSet.csv')
# end of Getting ready to create training data for ML Model

# Set of data that predictions will be used on[aka Test Data]
del companyDatasetLabeledVersion['Recommended']
companyDatasetLabeledVersion['Recommended'] = pd.Series((np.nan,np.nan,np.nan))
companyDatasetLabeledVersion.to_csv('DataPreparation/firstVersionTestSet.csv')

# end of set of data that prediction will be used on[aka Test Data]
# COMPLETE!!!!!
# NOTE: Will be forced to manually query for the numbers and then create a dataframe using these numbers.
# The following needs to be used to obtain the numbers for certain metrics: dataset1.index[1][1:] 

# Creating indices for new versions of datasets: 
# Applying the formulas above
  # Indexing each col, then referencing its value, by adding series by putting the nececssary number sin Series objects. 
#

# end of b)

# c) transforming dataframe into .csv file to be sent over to model. 


# end of c)

# end of body of code that brute forced data processing




# Section 1: 
# Purpose of section:

# end of purpose of section

# Psuedosteps for exercising purpose of section

# end of psuedosteps for exercising purpose of section 

# End of Section 1
