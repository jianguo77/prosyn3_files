# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 08:07:11 2020

@author: Nisha Haulkhory
"""
import csv, json
import pandas as pd



df_training= pd.DataFrame(pd.read_csv("df_training.csv", names=['emotion', 'pixels', 'Usage']))
df_training.to_json("df.json")
                                   
df = pd.read_json ('data.json') 
print(df.head())                             