#%%
import pandas as pd
from matplotlib.pyplot import figure, savefig, show, title
from ds_charts import bar_chart, get_variable_types
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from seaborn import heatmap

filename = '../drought.csv'
data = pd.read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)
data['date'] = pd.to_datetime(data['date'],format = '%d/%m/%Y')

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
savefig(f'imageD2/sparsity_study_all_vars.png',dpi=50)
show()

'''
fig = figure(figsize=[12, 12])
corr_mtx = abs(data.corr())

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues',annot_kws={"size": 4},fmt=".2f")
title('Correlation analysis')
savefig(f'imageD2/correlation_analysis.png')
show()'''

# %%
