from matplotlib.pyplot import subplots, figure, xticks, show, savefig
from pandas import read_csv, Series
from ts_functions import plot_series, HEIGHT
import matplotlib.pyplot as plt
from numpy import ones

data = read_csv('../drought.forecasting_dataset.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

index = data.index.to_period('W')
week_df = data.copy().groupby(index).sum()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('daily', fontweight="bold")
axs[0].text(0, 0, str(data.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('WEEKLY', fontweight="bold")
axs[1].text(0, 0, str(week_df.describe()))
show()

_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
data.boxplot(ax=axs[0])
week_df.boxplot(ax=axs[1])
show()
savefig("imagesD2Distribution/distribution_boxplot.png")

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for daily PRECTOT %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data['PRECTOT'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_daily_prectot.png")

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for daily PS %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data['PS'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_daily_PS.png")

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for daily T2M %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data['T2M'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_daily_T2M.png")

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for daily T2MDEW %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data['T2MDEW'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_daily_T2MDEW.png")

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for daily T2MWET %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data['T2MWET'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_daily_T2MWET.png")

bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for daily TS %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(data['TS'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_daily_TS.png")

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
    axs[j].set_title('Histogram for weekly PRECTOT %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df['PRECTOT'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_weekly_PRECTOT.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly PS %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df['PS'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_weekly_PS.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly T2M %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df['T2M'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_weekly_T2M.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly T2MDEW %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df['T2MDEW'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_weekly_T2MDEW.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly T2MWET %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df['T2MWET'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_weekly_T2MWET.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly TS %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df['TS'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_weekly_TS.png")

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for weekly QV2M %d bins'%bins[j])
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df['QV2M'], bins=bins[j])
show()
savefig("imagesD2Distribution/distribution_hist_weekly_QV2M.png")

# dt_series = Series(data['meter_reading'])

# mean_line = Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
# series = {'ashrae': dt_series, 'mean': mean_line}
# figure(figsize=(3*HEIGHT, HEIGHT))
# plot_series(series, x_label='timestamp', y_label='consumption', title='Stationary study', show_std=True)
# show()