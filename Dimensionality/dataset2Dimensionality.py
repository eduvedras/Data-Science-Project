#%%
import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types
from pandas import DataFrame


data = pd.read_csv("./drought.csv", na_values='', parse_dates=True, infer_datetime_format=True)
data['date'] = pd.to_datetime(data['date'],format = '%d/%m/%Y')

pd.plotting.register_matplotlib_converters()

data.shape

figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('imageDataset2/records_variables.png')
show()

variable_types = get_variable_types(data)
#print(variable_types)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('imageDataset2/variable_types.png')
show()
'''
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure()
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('imageDataset2/mv.png')
show()'''
# %%
