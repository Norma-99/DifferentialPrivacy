#This document does the preprocessing of the Adult NN and saves different .pickle: validation and training
import pandas as pd
import tensorflow as tf
import pickle

SIZE = 48842

def to_one_hot(df, col:str, exclude:set=None):
    if exclude is None:
        exclude = set()

    cs = set(df[col])
    #print(cs)
    for c in cs - exclude:
        df[c.lower()] = df[col] == c
        df[c.lower()] = df[c.lower()].astype(int)
    del df[col]

def norm_col(df, col:str):
    max_val = max(df[col])
    df[col] = df[col] / max_val

# Read CSV
df = pd.read_csv('datasets/dataset.csv')

# Get parameters to one-hot or remove them
df['age'] = df['age'] / 120.0

to_one_hot(df, 'workclass', {'?'})

del df['fnlwgt']
del df['education']
norm_col(df, 'education-num')

to_one_hot(df, 'occupation', {'?'})
#to_one_hot(df, 'marital-status')
del df['marital-status']

#to_one_hot(df, 'relationship')
del df['relationship']
to_one_hot(df, 'race')

df['sex'] = df['sex'] == 'Male'
df['sex'] = df['sex'].astype(int)

norm_col(df, 'capital-gain')
norm_col(df, 'capital-loss')
norm_col(df, 'hours-per-week')

to_one_hot(df, 'native-country', {'?'})

df['income'] = df['income'] == '>50K'
df['income'] = df['income'].astype(int)

x_dataset = df.copy()
del x_dataset['income']
y_dataset = df['income']

#Separate values in x and y
x, y = x_dataset.values, y_dataset.values
x = x / 255
print(y.shape)

#Separate values in train and validation
val_pair = x[0 : round(SIZE*0.1)], y[0 : round(SIZE*0.1)]
train_pair = x[round(SIZE*0.1) + 1 : round(SIZE*0.9)], y[round(SIZE*0.1) + 1 : round(SIZE*0.9)]

#Save test_dataset
with open('datasets/Mod_val/val_dataset.pickle', 'wb') as f:
    pickle.dump(val_pair, f)

#Save validation_dataset
with open('datasets/Mod_test/train_dataset.pickle', 'wb') as f:
    pickle.dump(train_pair, f)