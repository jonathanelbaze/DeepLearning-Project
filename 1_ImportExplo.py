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
# Train - set column names
train.columns = ['Image', 'Box 1', 'Box 2', 'Box 3', 'Box 4', 'Y']
# Add Columns for Maker, Model and Year categories
train['YMak'] = pd.Series(np.zeros(len(train), int))

train['YMod'] = pd.Series(np.zeros(len(train), int))

train['YYear'] = pd.Series(np.zeros(len(train), int))


test = pd.read_csv("anno_test.csv", header=None)
print(test.shape)

names = pd.read_csv("names.csv", header=None)
print(names.shape)


########################################             Exploration            ###########################################

#######    NAMES    ######

# Names - Extract Maker
names.columns = ['Key']
names['Maker'] = names['Key'].str.split(';').str[0]


# Names - Extract Year
names['Year'] = names['Key'].str.split().str[-1]

# Names - Extract Model
Model = names['Key'].str.split(';').str[1].str.split().str[0:-1]

ModelList = []
for e in Model:
    ModelList.append(''.join(str(a)+' ' for a in e))


names['Model'] = ModelList



######      Convert Old Y to new categories      ######
Makers = sorted(list(set(names['Maker'])))

MakersID = []
for i in names['Maker']:
    MakersID.append(Makers.index(i))


for idx, car in train['Y'].iteritems():
    train['YMak'][idx] = MakersID[car-1]

print(train)

np.savetxt(r'trainMaker.txt', train['YMak'].values, fmt='%d')

# MakersList=[]
# MakersDict = {}
# for idx, car in names['Maker'].iteritems():
#     if car not in MakersDict.keys():
#         MakersDict[car] = []
#     MakersDict[car].append(idx)

    # if car in MakersDict.keys():
    # MakersDict[car].add(idx)





# print(train)

# for idx, row in train.iterrows():
#     for maker in MakersDict:
#         if row[5] in MakersDict[maker]:
#             train['YMak',idx] = maker
#
# print(train)
