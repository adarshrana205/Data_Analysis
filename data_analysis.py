#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:03:42 2019

@author: adarshrana205
"""

import pandas as pd
import numpy as np
path="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

headers=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
df=pd.read_csv(path,header=None)
df.columns=headers
print(df.head(5))
print(df.dtypes)
print(df.describe(include="all"))
df["a"]=df["a"]+1
print(df.head(5))
df.dropna(subset=["price"],axis=0,inplace=True)
mean = df["a"].mean() 
df["a"].replace(np.nan, mean)


# Data Normalization
df["b"]=df["b"]/df["b"].max() # Simple feature scaling method
df["b"]=(df["b"]-df["b"].min())/(df["b"].max()-df["b"].min()) # Min-max method
df["b"]=(df["b"]-df["b"].mean())/df["b"].std()   # Z-score method

#Data Binning
bins=np.linspace(min(df["b"]),max(df["b"]),4)
group_names=["Low","Medium","High"]
df["b-binned"]=pd.cut(df["b"],bins,labels=group_names,include_lowest=True)
