#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:03:42 2019
@author: adarshrana205
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import pipeline 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

path="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers=["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]
df=pd.read_csv(path,header=None) #import data	
# To export data we use "to_csv" method
df.columns=headers
print(df.head(5))
print(df.dtypes)

df["symboling"]=df["symboling"]+1
print(df.head(5))

# How to deal with missing values in data
df.dropna(subset=["price"],axis=0,inplace=True)
mean = df["symboling"].mean() 
df["symboling"].replace(np.nan, mean)


# Data Normalization
df["normalized-losses"]=df["normalized-losses"]/df["normalized-losses"].max() # Simple feature scaling method
df["normalized-losses"]=(df["normalized-losses"]-df["normalized-losses"].min())/(df["normalized-losses"].max()-df["normalized-losses"].min()) # Min-max method
df["normalized-losses"]=(df["normalized-losses"]-df["normalized-losses"].mean())/df["normalized-losses"].std()   # Z-score method

#Data Binning
bins=np.linspace(min(df["normalized-losses"]),max(df["normalized-losses"]),4)
group_names=["Low","Medium","High"]
df["normalized-losses-binned"]=pd.cut(df["normalized-losses"],bins,labels=group_names,include_lowest=True)

#Turning categorical variables into quantitative variables
pd.get_dummies(df['fuel-type'])

#Exploratory Data Analysis
print(df.describe(include="all")) # check statiscal summary
# For categorical variables 
drive_wheels_counts=df["drive-wheels"].value_counts()

drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'},inplace=True)
drive_wheels_counts.index.name='drive-wheels'

#Box Plot for numeric values
sns.boxplot(x="drive-wheels",y="price",data=df)

#scatter plot of continuous variables to show relationship between two variables
y=df["price"]#Dependent variable
x=df["engine-size"]#Independent variable
plt.scatter(x,y)
plt.title("Scatterplot of engine-size and price")
plt.xlabel("Engine Size")
plt.ylabel("Price")


#GroupBy
df_test=df['drive-wheels','body-style','price']
df_grp=df_test.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(df_grp)

#Easier to read we use use pivot table
df_pivot=df_grp.pivot(index='drive-wheels',columns='body-style')
#Another way to represent pivot table is using heatmap plot
plt.pcolor(df_pivot,cmap='RdBBu')
plt.colorbar()
plt.show()

#Correlation
sns.regplot(x="engine-size",y="price",data=df)
plt.ylim(0,) #positive correlation


sns.regplot(x="highway-mpg",y="price",data=df)
plt.ylim(0,) #negative correlation

sns.regplot(x="peak-rpm",y="price",data=df)
plt.ylim(0,) #weak correlation

#Correlation Statistical methods

#Pearson Correlation(measure strength of correlation between two features and gives two values 1:)Correlation coefficient 2:)P-value )
Pearson_coef,p_value=stats.pearsonr[['horsepower'],df['price']]


#ANOVA(Analysis of variance used to find the correlation between different groups of a categorical variable
#  and return two values the F-test score and the p-value )
df_anova=df[["make","price"]]
grouped_anova=df_anova.groupby["make"]
anova_results_1=stats.f_oneway(grouped_anova.get_group("honda")["price"],grouped_anova.get_group("subaru")["price"])
#(Small F value)

anova_results_1=stats.f_oneway(grouped_anova.get_group("honda")["price"],grouped_anova.get_group("jaguar")["price"])
#(Large F value i.e. strong correlation between a categorical variable and other variables )


#Simple linear regression
lm=LinearRegression()
X=df[['highway-mpg']]
Y=df[['price']]
lm.fit(X,Y)
Yhat=lm.predict(X)

#Multiple linear regression
z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
lm.fit(z,df['price'])
Yhat=lm.predict(z)


#Model Evaluation using Visualization
sns.regplot(x="highway-mpg",y="price",data=df)
plt.ylim(0,)

sns.residplot(df['highway-mpg'],df['price'])#Residual plot(residual plot represents the error between the actual value)

ax1=sns.distplot(df['price'],hist=False,color='r',label='Actual Value')#Distribution Plot(distribution plot counts the predicted value versus the actual value)
sns.distplot(Yhat,hist=False,color='b',label='Fitted Values',ax=ax1)


#Polynimial Regression
f=np.polyfit(x,y,3)
p=np.polydl(f)
print(p)



#Polynimial Regression with more than one dimension(Numpy's polyfit function cannot perform this type of regression)
pr=PolynomialFeatures(degree=2)
x_polly=pr.fit_transform(x[['horsepower','curb-weight']],include_bias=false)


#If Dimension gets larger
SCALE=StandardScaler()
SCALE.fit(x_data[['horsepower','highway-mpg']])
x_scale=SCALE.transform(x_data[['horsepower','highway-mpg']])


#Pipeline
Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('mode',LinearRegression())]
pipe=pipeline(Input)
pipe.train(X['horsepower','curb-weight','engine-size','highway-mpg'])
yhat=pipe.predict(X[['horsepower','curb-weight','engine-size','highway-mpg']])

#Mean Squared Error(MSE)
mean_squared_error(df['price'],Y_predict_simple_fit)

#R-squared
X=df[['highway-mpg']]
Y=df[['price']]
lm.fit(X,Y)
lm.score(X,Y)

#Splitting training and testing data
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.3,random_state=0)

#Cross Validations
scores=cross_val_score(lm,x_data,y_data,cv=3)
np.mean(scores)
#For prediction(actual predicted values)
yhat=cross_val_predict(lm,x_data,y_data,cv=3)


#Different r-squared values
Rsqu_test=[]
order=[1,2,3,4]
for n in order:
    pr=PolynomialFeatures(degree=n)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    x_test_pr=pr.fit_transform(x_test[['horsepower']])
    lm.fit(x_train_pr,y_train)
    Rsqu_test.append(lm.score(x_test_pr,y_test))


























