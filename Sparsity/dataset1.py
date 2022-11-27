#%%
import pandas as pd
from matplotlib.pyplot import figure, savefig, show, title
from ds_charts import bar_chart, get_variable_types
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from seaborn import heatmap

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
savefig(f'imageD1/sparsity_study_all_vars.png')
show()


fig = figure(figsize=[12, 12])
corr_mtx = abs(data.corr())

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig(f'imageD1/correlation_analysis.png')
show()
# %%
