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


def create_list(df):
    #Create a list from a column dataframe
    l = []
    for i in df:
        if not i in l:
            l.append(i)
    return l

print(create_list(names['Maker']))

print(len(create_list(names['Maker'])))

print(create_list(names['Model']))

print(len(create_list(names['Model'])))

def add_values_Y(datalist, nameslist, dataframe, column):
# Add new Y values to Train or Test Dataframe (YMak or YMod or YYear)
    IDlist = []
    for i in nameslist:
        IDlist.append(datalist.index(i))

    for idx, car in dataframe['Y'].iteritems():
        dataframe[column][idx] = IDlist[car-1]


add_values_Y(create_list(names['Maker']), names['Maker'], train, 'YMak')
add_values_Y(create_list(names['Model']), names['Model'], train, 'YMod')
add_values_Y(create_list(names['Year']), names['Year'], train, 'YYear')
print(train)

# print(train)

train_maker_labels = train['YMak'].to_numpy
train_model_labels = train['YMod'].to_numpy
train_year_labels = train['YYear'].to_numpy
# print(train_maker_labels)
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
