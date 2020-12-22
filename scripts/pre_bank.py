#This document does the preprocessing of the Banking Dataset and saves different .pickle: validation and training
import pandas as pd
import tensorflow as tf
import pickle

SIZE = 45211

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
df = pd.read_csv('datasets/bank/bank-full.csv')

# Get parameters to one-hot or remove them
df['age'] = df['age'] / 120.0

to_one_hot(df, 'job', {'?'})
to_one_hot(df, 'marital', {'?'}) # del df['marital-status']
to_one_hot(df, 'education', {'?'})
to_one_hot(df, 'default', {'?'})
norm_col(df, 'balance')
to_one_hot(df, 'housing', {'?'})
to_one_hot(df, 'loan', {'?'})
del df['contact']
del df['day']
to_one_hot(df, 'month', {'?'})
del df['duration']
del df['campaign']
norm_col(df, 'pdays')
norm_col(df, 'previous')
to_one_hot(df, 'poutcome', {'?'}) 

df['y'] = df['y'] == 'yes'
df['y'] = df['y'].astype(int)

x_dataset = df.copy()
del x_dataset['y']
y_dataset = df['y']

#Separate values in x and y
x, y = x_dataset.values, y_dataset.values
print(y.shape)
print(df.columns)

#Separate values in train and validation
val_pair = x[0 : round(SIZE*0.1)], y[0 : round(SIZE*0.1)]
train_pair = x[round(SIZE*0.1) + 1 : round(SIZE*0.9)], y[round(SIZE*0.1) + 1 : round(SIZE*0.9)]

#Save test_dataset
with open('datasets/bank/val_dataset.pickle', 'wb') as f:
    pickle.dump(val_pair, f)

#Save validation_dataset
with open('datasets/bank/train_dataset.pickle', 'wb') as f:
    pickle.dump(train_pair, f)