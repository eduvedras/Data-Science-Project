from matplotlib.pyplot import subplots, figure, xticks, show, savefig
from pandas import read_csv, Series
from ts_functions import plot_series, HEIGHT
import matplotlib.pyplot as plt
from numpy import ones

data = read_csv('../droughtDrop.csv', index_col='date', sep=',', decimal='.', parse_dates=True, dayfirst=True)

index = data.index.to_period('M')
month_df = data.copy().groupby(index).sum()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)
index = data.index.to_period('Q')
quarter_df = data.copy().groupby(index).mean()
quarter_df['timestamp'] = index.drop_duplicates().to_timestamp()
quarter_df.set_index('timestamp', drop=True, inplace=True)
_, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('daily', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('Monthly', fontweight="bold")
axs[1].text(0, 0, str(month_df.describe()))
axs[2].grid(False)
axs[2].set_axis_off()
axs[2].set_title('QUarterly', fontweight="bold")
axs[2].text(0, 0, str(quarter_df.describe()))
show()

_, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT))
data.boxplot(ax=axs[0])
month_df.boxplot(ax=axs[1])
quarter_df.boxplot(ax=axs[2])
show()
savefig("imagesD2Distribution/distribution_boxplot.png")

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for daily QV2M %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data['QV2M'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_daily_QV2M.png")

##
##-----------Weekly hists
##

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for monthly QV2M %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(month_df['QV2M'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_monthly_QV2M.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for quarterly QV2M %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(quarter_df['QV2M'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_quarterly_QV2M.png")

# dt_series = Series(data['meter_reading'])

# mean_line = Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
# series = {'ashrae': dt_series, 'mean': mean_line}
# figure(figsize=(3*HEIGHT, HEIGHT))
# plot_series(series, x_label='timestamp', y_label='consumption', title='Stationary study', show_std=True)
# show()