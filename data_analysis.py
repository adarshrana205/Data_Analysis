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

ax1=sns.distplot(df['price'],hist=False,color='r',label='Actual Value')#Distribution Plo(distribution plot counts the predicted value versus the actual value)
sns.distplot(Yhat,hist=False,color='b',label='Fitted Values',ax=ax1)

#Polynimial Regression
f=np.polyfit(x,y,3)
p=np.polydl(f)
print(p)

#Polynimial Regression with more than one dimension
pr=PolynomialFeatures(degree=2)
x_polly=pr.fit_transform(x[['horsepower','curb-weight']],include_bias=false)





























