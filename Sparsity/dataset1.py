#%%
import pandas as pd
from matplotlib.pyplot import figure, savefig, show, title
from ds_charts import bar_chart, get_variable_types
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from seaborn import heatmap
from sklearn.preprocessing import OneHotEncoder
from numpy import number
from pandas import DataFrame, concat

filename = '../diabetic_data.csv'
data = pd.read_csv(filename)
'''
numeric_vars = get_variable_types(data)['Numeric']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(numeric_vars)):
    var1 = numeric_vars[i]
    for j in range(i+1, len(numeric_vars)):
        var2 = numeric_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'imageD1/sparsity_study_numeric.png')
show()

symbolic_vars = get_variable_types(data)['Symbolic']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(symbolic_vars)):
    var1 = symbolic_vars[i]
    for j in range(i+1, len(symbolic_vars)):
        var2 = symbolic_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'imageD1/sparsity_study_symbolic.png')
show()

binary_vars = get_variable_types(data)['Binary']
if [] == binary_vars:
    raise ValueError('There are no binary variables.')

rows, cols = len(binary_vars)-1, len(binary_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(binary_vars)):
    var1 = binary_vars[i]
    for j in range(i+1, len(binary_vars)):
        var2 = binary_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'imageD1/sparsity_study_binary.png')
show()'''

all_vars = list(data.columns)
if [] == all_vars:
    raise ValueError('There are no all variables.')

rows, cols = len(all_vars)-1, len(all_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(all_vars)):
    var1 = all_vars[i]
    for j in range(i+1, len(all_vars)):
        var2 = all_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'imageD1/sparsity_study_all_vars.png',dpi=50)
show()

'''
def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df


filename = '../diabetic_data.csv'
data = pd.read_csv(filename, na_values='?')
data.dropna(inplace=True)

variables = get_variable_types(data)
symbolic_vars = variables['Symbolic']
symbolic_vars.remove('readmitted')
print(symbolic_vars)
data.readmitted = data.readmitted.replace('NO','0')
data.readmitted = data.readmitted.replace('<30','1')
data.readmitted = data.readmitted.replace('>30','2')
#print(data.to_string())

df = dummify(data, symbolic_vars)

filename = 'data/diabetic_dummified1.csv'
df = pd.read_csv(filename)

df = df.replace('No','0')
df = df.replace('Yes','1')
df = df.replace('Male','0')
df = df.replace('Female','1')
df = df.replace('Steady','1')
df = df.replace('Ch','1')
df = df.replace(False,'0')
df = df.replace(True,'1')
df.to_csv(f'data/diabetic_data_dummified.csv', index=False)




filename = 'data/diabetic_data_dummified.csv'
data = pd.read_csv(filename)
fig = figure(figsize=[60, 60])
corr_mtx = abs(data.corr(numeric_only=False))
print(corr_mtx)

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues',annot_kws={"size": 1},fmt=".2f")
title('Correlation analysis')
savefig(f'imageD1/correlation_analysis.png')
show()'''
# %%
