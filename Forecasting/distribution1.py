from matplotlib.pyplot import subplots, figure, xticks, show, savefig
from pandas import read_csv, Series
from ts_functions import plot_series, HEIGHT
import matplotlib.pyplot as plt
from numpy import ones

data = read_csv('../glucose.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, dayfirst=True)

index = data.index.to_period('H')
hour_df = data.copy().groupby(index).sum()
hour_df['timestamp'] = index.drop_duplicates().to_timestamp()
hour_df.set_index('timestamp', drop=True, inplace=True)
index = data.index.to_period('D')
day_df = data.copy().groupby(index).sum()
day_df['timestamp'] = index.drop_duplicates().to_timestamp()
day_df.set_index('timestamp', drop=True, inplace=True)
index = data.index.to_period('W')
week_df = data.copy().groupby(index).sum()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
_, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('HOURLY', fontweight="bold")
axs[0].text(0, 0, str(hour_df.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('DAILY', fontweight="bold")
axs[1].text(0, 0, str(day_df.describe()))
axs[2].grid(False)
axs[2].set_axis_off()
axs[2].set_title('WEEKLY', fontweight="bold")
axs[2].text(0, 0, str(week_df.describe()))
show()

_, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT))
hour_df.boxplot(ax=axs[0])
day_df.boxplot(ax=axs[1])
week_df.boxplot(ax=axs[2])
show()
savefig("imagesD1Distribution/distribution_boxplot.png")

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for hourly glucose %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(hour_df['Glucose'], bins=bins[j])
show()
savefig("imagesD1Distribution/distribution_hist_hourly_glucose.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for daily glucose %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(day_df['Glucose'], bins=bins[j])
show()
savefig("imagesD1Distribution/distribution_hist_daily_glucose.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly glucose %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df['Glucose'], bins=bins[j])
show()
savefig("imagesD1Distribution/distribution_hist_weekly_glucose.png")
