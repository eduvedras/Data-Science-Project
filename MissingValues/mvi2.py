from pandas import read_csv, to_datetime
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types
from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame
from numpy import nan


register_matplotlib_converters()
file = 'drought_wdate'
filename = '../drought.csv'
data = read_csv("../drought.csv", parse_dates=True, infer_datetime_format=True)
data['date'] = to_datetime(data['date'],format = '%d/%m/%Y')
#1st approach
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data = data.drop(['date'],axis=1)
data.to_csv(f'data/{file}.csv', index=False)

#2nd approach
#data = data.drop(['date'],axis=1)
#data.to_csv(f'data/{file}.csv', index=False)