# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:45:53 2020

@author: Georges & Jonathan
"""


# -*- coding: utf-8 -*-
"""
Importing the data and exploring it
"""
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

#import seaborn as sns

#import difflib

########################################        Printing Settings           ###########################################

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


########################################           Import Data              ###########################################
# Set working directory
path = os.getcwd()

# Import
train = pd.read_csv("anno_train.csv", header=None)
print(train.shape)

test = pd.read_csv("anno_test.csv", header=None)
print(test.shape)

names = pd.read_csv("names.csv", header=None)
print(names.shape)


########################################             Exploration            ###########################################
print(train.head())

# Train - set column names
train.columns = ['Image', 'Box 1', 'Box 2', 'Box 3', 'Box 4', 'Y']
print(train.head())

# Names - Extract Maker
names.columns = ['Key']
names['Maker'] = names['Key'].str.split(';').str[0]


# names['Model']
Model = names['Key'].str.split(';').str[1].str.split().str[0:-1]

ModelList = []
for e in Model:
    ModelList.append(''.join(str(a)+' ' for a in e))


names['Model'] = ModelList

# print(ModelList)



# Names - Extract Year
names['Year'] = names['Key'].str.split().str[-1]
print(names['Year'])
# Names - Extract Model
info=names['Key']
info = info.to_frame()
info.columns = ['Year']

Models = set(ModelList)
print(Models)

Makers = set(names['Maker'])

MakersDict = {}
for idx, car in names['Maker'].iteritems():
    print(idx, car)
    # if car in MakersDict.keys():
    # MakersDict[car].add(idx)

print(MakersDict)

