##################################################################
# Data
"""
path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
"""
##################################################################

"""

Creating train and test dataset
Train/Test Split involves splitting the dataset into training and testing sets that are mutually exclusive. 
After which, you train with the training set and test with the testing set. 
This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset 
is not part of the dataset that have been used to train the model. Therefore, it gives us a better understanding 
of how well our model generalizes on new data.

This means that we know the outcome of each data point in the testing dataset, 
making it great to test with! Since this data has not been used to train the model, 
the model has no knowledge of the outcome of these data points. 
So, in essence, it is truly an out-of-sample testing.

Let's split our dataset into train and test sets. 
80% of the entire dataset will be used for training and 20% for testing. 
We create a mask to select random rows using np.random.rand() function:

"""

##################################################################

# Import

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import matplotlib as mpt
from sklearn.neighbors import LocalOutlierFactor

mpt.use('Qt5Agg')

# Setting
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

df = pd.read_csv("datasets/IBM/FuelConsumptionCo2.csv")

df.columns
df.head()
"""
Index(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 
'ENGINESIZE', 'CYLINDERS', 'TRANSMISSION', 'FUELTYPE', 
'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 
'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS'], dtype='object')
"""

df.isnull().any()

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

cdf.head()

cdf.info

sns.regplot(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue', line_kws={"color": "r"})
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("ENGINESIZE and CO2EMISSIONS Scattering")
plt.show()

sns.scatterplot(x=cdf.ENGINESIZE, y=cdf.CO2EMISSIONS)

msk = np.random.rand(len(df)) < 0.80
len(msk)
train = cdf[msk]
train.shape
train.head()
test = cdf[~msk]
test.head()
test.shape

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_predict = regr.predict(test_x)
test_predict[0:5]
type(test_predict)
type(test_y)

test_y_list = pd.DataFrame(test_y.tolist())
test_predict_list = pd.DataFrame(test_predict.tolist())

result = pd.concat([test_y_list, test_predict_list], axis=1)
result.columns = ["CO2EMISSIONS", "Prediction"]
result["Error"] = result["CO2EMISSIONS"] - result["Prediction"]
result["Error-square"] = result["Error"]**2

result.head(15)
result.describe().T

from sklearn.metrics import r2_score
print("R2-score: %.3f" % r2_score(test_y , test_predict))

# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ', regr.intercept_)

# regr.coef_ çıktısı array([39.63100602])
#   regr.coef_[0][0] dediğimizde içindeki sayıyı alıyor

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Train Data Set Plot")

from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_predict - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_predict - test_y) ** 2))
print("R2-score: %.3f" % r2_score(test_y , test_predict))

df[(df["MAKE"] == "VOLKSWAGEN") &
   (df["FUELTYPE"] == "Z")].groupby("VEHICLECLASS").agg({"FUELCONSUMPTION_CITY": "mean",
                                                         "FUELCONSUMPTION_HWY": "mean",
                                                         "FUELCONSUMPTION_COMB": "mean"})

df["FUELTYPE"].unique()

"""
Z: premium gasoline - more than 91 octan benzine
D: diesel
X: reguler gasoline - below 91 octan
E: E85 - mix gasoline with etile alcohol
"""