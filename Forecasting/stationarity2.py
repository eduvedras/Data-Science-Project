from numpy import ones
from pandas import Series,read_csv
from pandas import read_csv
from matplotlib.pyplot import figure, xticks, show, savefig
import matplotlib.pyplot as plt
from ts_functions import plot_series, HEIGHT

data = read_csv('../drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)

dt_series = Series(month_df['QV2M'])

mean_line = Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
series = {'QV2M': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='timestamp', y_label='QV2M', title='Stationary study', show_std=True)
show()
savefig("imagesD2Stationarity/QV2M.png")

BINS = 10
line = []
n = len(dt_series)
for i in range(BINS):
    b = dt_series[i*n//BINS:(i+1)*n//BINS]
    mean = [b.mean()] * (n//BINS)
    line += mean
line += [line[-1]] * (n - len(line))
mean_line = Series(line, index=dt_series.index)
series = {'QV2M': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='time', y_label='QV2M', title='Stationary study', show_std=True)
show()
savefig("imagesD2Stationarity/QV2M_all.png")