#%%
from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types
from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame
from numpy import nan


register_matplotlib_converters()
file = 'glucoseDrop'
filename = '../glucose.csv'
data = read_csv(filename, na_values='')

data = data.drop('Insulin', axis=1)
data.to_csv(f'../{file}.csv', index=False)

# %%
