#%%
import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

filename = '../drought.csv'
data = pd.read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)
data['date'] = pd.to_datetime(data['date'],format = '%d/%m/%Y')