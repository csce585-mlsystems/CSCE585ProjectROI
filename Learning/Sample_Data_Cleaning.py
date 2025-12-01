import pandas as pd
from io import StringIO
#String IO allows us to read the string assigned to csv_data

csv_data = \
    '''A, B, C, D
 1.0, 2.0,3.0,4.0'
 5.0, 6.0, 8.0'
 10.0, 11.0, 12.0'''

df = pd.read_csv(StringIO(csv_data))

#--------------------------------------Removeing missing values------------------------------------------------------#
df.isnull().sum()

#Remove training (rows) or features (columns)
df.dropna(axis=0)  #removes NaN rows

df.dropna(axis=1) #removes NaN columns

df.dropna(how='all')  #removes columns that have NaN 

df.dropna(thresh=4) #removes rows that have fewer than 4 values

df.dropna(subset=["C"]) #remove NAN rows within column C

#----------------------------------Imputing Missiong Values---------------------------------------------------

#Here we are using the mean imputation technqiue where we replace NAN values with the mean

from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy = "mean")  #We could also use media or most frequent
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data