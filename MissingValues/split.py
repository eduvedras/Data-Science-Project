import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split

file_tag = 'diabetic_data_100'
data: DataFrame = read_csv('data/constant_dummified1.csv')
data = data.iloc[0:100,:]
target = 'readmitted'
no = 'NO'
above30 = '>30'
under30 = '<30'
values = {'Original': [len(data[data[target] == above30]), len(data[data[target] == under30]), len(data[data[target] == no])]}

y: np.ndarray = data.pop(target).values
X: np.ndarray = data.values
labels: np.ndarray = unique(y)
labels.sort()

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
train.to_csv(f'data/{file_tag}_train.csv', index=False)

test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
test.to_csv(f'data/{file_tag}_test.csv', index=False)
values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==above30))), len(np.delete(trnY, np.argwhere(trnY==under30))), len(np.delete(trnY, np.argwhere(trnY==no)))]
values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==above30))), len(np.delete(tstY, np.argwhere(tstY==under30))), len(np.delete(trnY, np.argwhere(trnY==no)))]