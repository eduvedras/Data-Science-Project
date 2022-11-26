#%%
import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types
from pandas import DataFrame


data = pd.read_csv("../diabetic_data.csv", na_values='?')

pd.plotting.register_matplotlib_converters()

data.shape

figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('imageDataset1/records_variables.png')
show()

variable_types = get_variable_types(data)
#print(variable_types)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('imageDataset1/variable_types.png')
show()

mv = {}
for var in data:
    #nr = len(data[data[var] == '?'])
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure(figsize=(4,10))
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('imageDataset1/mv.png')
show()
# %%
