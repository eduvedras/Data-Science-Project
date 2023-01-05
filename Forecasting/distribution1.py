from matplotlib.pyplot import subplots, figure, xticks, show, savefig
from pandas import read_csv, Series
from ts_functions import plot_series, HEIGHT
import matplotlib.pyplot as plt
from numpy import ones

data = read_csv('../glucose.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

index = data.index.to_period('M')
month_df = data.copy().groupby(index).sum()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)
index = data.index.to_period('Q')
qua_df = data.copy().groupby(index).sum()
qua_df['timestamp'] = index.drop_duplicates().to_timestamp()
qua_df.set_index('timestamp', drop=True, inplace=True)
_, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('HOURLY', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('MONTHLY', fontweight="bold")
axs[1].text(0, 0, str(month_df.describe()))
axs[2].grid(False)
axs[2].set_axis_off()
axs[2].set_title('QUATERLY', fontweight="bold")
axs[2].text(0, 0, str(qua_df.describe()))
show()

_, axs = subplots(1, 3, figsize=(2*HEIGHT, HEIGHT))
data.boxplot(ax=axs[0])
month_df.boxplot(ax=axs[1])
qua_df.boxplot(ax=axs[2])
show()
savefig("imagesD1Distribution/distribution_boxplot.png")

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for hourly glucose %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data['Glucose'], bins=bins[j])
show()
savefig("imagesD1Distribution/distribution_hist_hourly_glucose.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly glucose %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(month_df['Glucose'], bins=bins[j])
show()
savefig("imagesD1Distribution/distribution_hist_monthly_glucose.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly glucose %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(qua_df['Glucose'], bins=bins[j])
show()
savefig("imagesD1Distribution/distribution_hist_quaterly_glucose.png")
