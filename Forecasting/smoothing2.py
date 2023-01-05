from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots
from ts_functions import HEIGHT, split_dataframe, create_temporal_dataset
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES,HEIGHT, plot_evaluation_results, plot_forecasting_series
from statsmodels.tsa.arima.model import ARIMA
from ds_charts import multiple_line_chart

file_tag = 'drought_smoothing50'
index_multi = 'date'
target_multi = 'QV2M'
data_multi = read_csv('../drought.forecasting_dataset.csv', index_col=index_multi, parse_dates=True, dayfirst=True)

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

agg_multi_df = aggregate_by(data_multi, index_multi, 'M')

WIN_SIZE = 50
rolling_multi = agg_multi_df.rolling(window=WIN_SIZE)
smooth_df_multi = rolling_multi.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df_multi[target_multi], title=f'Drought - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='QV2M')
#plot_series(smooth_df_multi['PRECTOT'])
xticks(rotation = 45)
show()
savefig(f'imagesD2Transformation/{file_tag}.png')

df = smooth_df_multi.drop(['PRECTOT','PS','T2M','T2MDEW','T2MWET','TS'], axis=1)
df.drop(index=df.index[:WIN_SIZE], axis=0, inplace=True)
#df.to_csv(f'../{file_tag}.csv', index=False)


def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: DataFrame = df_cp.iloc[:trn_size, :]
    test: DataFrame = df_cp.iloc[trn_size:]
    return train, test

train, test = split_dataframe(df, trn_pct=0.75)

measure = 'R2'
flag_pct = False
eval_results = {}
last_best = -100
best = ('',  0, 0.0)
best_model = None

d_values = (0, 1, 2)
params = (1, 2, 3, 5)
ncols = len(d_values)

#fig, axs = subplots(1, ncols, figsize=(ncols*HEIGHT, HEIGHT), squeeze=False)

'''for der in range(len(d_values)):
    d = d_values[der]
    values = {}
    for q in params:
        yvalues = []
        for p in params:
            pred = ARIMA(train, order=(p, d, q))
            model = pred.fit(method_kwargs={'warn_convergence': False})
            prd_tst = model.forecast(steps=len(test), signal_only=False)
            yvalues.append(PREDICTION_MEASURES[measure](test,prd_tst))
            if yvalues[-1] > last_best:
                best = (p, d, q)
                last_best = yvalues[-1]
                best_model = model
        values[q] = yvalues
    multiple_line_chart(
        params, values, ax=axs[0, der], title=f'ARIMA d={d}', xlabel='p', ylabel=measure, percentage=flag_pct)
savefig(f'imagesD2ModelEval/{file_tag}_ts_arima_study.png')
show()
print(f'Best results achieved with (p,d,q)=({best[0]}, {best[1]}, {best[2]}) ==> measure={last_best:.2f}')'''

pred = ARIMA(train, order=(5, 1, 5))
best_model = pred.fit(method_kwargs={'warn_convergence': False})
prd_trn = best_model.predict(start=0, end=len(train)-1)
prd_tst = best_model.forecast(steps=len(test))
print(f'\t{measure}={PREDICTION_MEASURES[measure](test, prd_tst)}')

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD2ModelEval/{file_tag}_arima_5_1_5_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD2ModelEval/{file_tag}_arima_5_1_5_plots.png', x_label=index_multi, y_label=target_multi)

'''class PersistenceRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: DataFrame):
        self.last = X.iloc[-1,0]
        print(self.last)

    def predict(self, X: DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd

fr_mod = PersistenceRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['Persistence'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD2Transformation/{file_tag}_persistence_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD2Transformation/{file_tag}_persistence_plots.png', x_label=index_multi, y_label=target_multi)'''

'''class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: DataFrame):
        self.mean = X.mean()

    def predict(self, X: DataFrame):
        prd =  len(X) * [self.mean]
        return prd

fr_mod = SimpleAvgRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD2Transformation/{file_tag}_simpleAvg_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD2Transformation/{file_tag}_simpleAvg_plots.png', x_label=index_multi, y_label=target_multi)'''

'''class RollingMeanRegressor (RegressorMixin):
    def __init__(self, win: int = 2):
        super().__init__()
        self.win_size = win

    def fit(self, X: DataFrame):
        None

    def predict(self, X: DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd

fr_mod = RollingMeanRegressor()
fr_mod.fit(train)
prd_trn = fr_mod.predict(train)
prd_tst = fr_mod.predict(test)

eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
print(eval_results)

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD2ModelEval/{file_tag}_rollingMean_eval_2.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD2ModelEval/{file_tag}_rollingMean_plots_2.png', x_label=index_multi, y_label=target_multi)'''