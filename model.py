#generic libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/Users/vides/OneDrive/Documents/housing.csv")

print(df.info())
df=pd.get_dummies(df, columns=["ocean_proximity"])
print(df.head())
df["total_bedrooms"]=df["total_bedrooms"].fillna(df["total_bedrooms"].mean())
print(df.isnull().sum())
df.columns = df.columns.str.replace(" ", "_")

print("statistical data ")
print(df['households'].describe())


# sns.scatterplot(x=df["housing_median_age"],y=df["median_house_value"])
# plt.xlabel("Median Income")
# plt.ylabel("Median House Value")
# plt.title("Correlation Between Income and House Value")
# plt.show()

# # Numerical correlation value
# correlation = df["median_income"].corr(df["median_house_value"])

# print("Correlation:", correlation)

x=df.drop("median_house_value", axis=1)
y=df["median_house_value"]

from sklearn.model_selection import train_test_split

X_train, X_test ,  y_train , y_test=train_test_split(x,y, test_size=0.2, random_state=42)

#scalling=> to make all the columns in a range 
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

 # converting into dataframe
X_train = pd.DataFrame(
    X_train,
    columns=x.columns
)

X_test = pd.DataFrame(
    X_test,
    columns=x.columns
)

#model trainning
# from sklearn.linear_model import LinearRegression

# model=LinearRegression()



# using lightgbm 

import lightgbm as lgb 
model= lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=35,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print("this is the price of your house ", y_pred)

# evaluate model 

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# print("MAE: ", mean_absolute_error(y_test, y_pred))
# print("MSE : ", mean_squared_error(y_test, y_pred))
# print("R2 SCORE :", r2_score(y_test, y_pred))

# # compare actual vs predicted 
# comparison = pd.DataFrame({
#     "actual:" :y_test,
#     "Predicted:": y_pred
# })
# print(comparison.head())

import pickle

pickle.dump(
    model,
    open("lightgbm_model.pkl", "wb")
)

pickle.dump(
    scaler,
    open("scaler.pkl", "wb")
)

print("Files Saved Successfully")
