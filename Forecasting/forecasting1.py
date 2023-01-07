from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots
from ts_functions import HEIGHT, split_dataframe, create_temporal_dataset
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series


file_tag = 'glucose_forecasting'
index_multi = 'Date'
target_multi = 'Glucose'
data_multi = read_csv('../glucose.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, dayfirst=True)

train, test = split_dataframe(data_multi, trn_pct=0.75)

measure = 'R2'
flag_pct = False
eval_results = {}

class PersistenceRegressor (RegressorMixin):
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

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD1Forecasting/{file_tag}_persistence_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD1Forecasting/{file_tag}_persistence_plots.png', x_label=index_multi, y_label=target_multi)


class SimpleAvgRegressor (RegressorMixin):
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

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD1Forecasting/{file_tag}_simpleAvg_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD1Forecasting/{file_tag}_simpleAvg_plots.png', x_label=index_multi, y_label=target_multi)
'''
class RollingMeanRegressor (RegressorMixin):
    def __init__(self, win: int = 10):
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

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD1Forecasting/{file_tag}_win=10_rollingMean_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD1Forecasting/{file_tag}_win=10_rollingMean_plots.png', x_label=index_multi, y_label=target_multi)
'''

from statsmodels.tsa.arima.model import ARIMA

#pred = ARIMA(train, order=(2, 0, 2))
#model = pred.fit(method_kwargs={'warn_convergence': False})
#model.plot_diagnostics(figsize=(2*HEIGHT, 2*HEIGHT))

from matplotlib.pyplot import subplots, show, savefig
from ds_charts import multiple_line_chart
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

measure = 'R2'
flag_pct = False
last_best = -100
best = ('',  0, 0.0)
best_model = None

d_values = (0, 1, 2)
params = (1, 2, 3, 5)
ncols = len(d_values)
'''
fig, axs = subplots(1, ncols, figsize=(ncols*HEIGHT, HEIGHT), squeeze=False)

for der in range(len(d_values)):
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
savefig(f'imagesD1Arima/{file_tag}_ts_arima_study.png')
show()
print(f'Best results achieved with (p,d,q)=({best[0]}, {best[1]}, {best[2]}) ==> measure={last_best:.2f}')
'''

from statsmodels.tsa.arima.model import ARIMA
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

md = ARIMA(train, order=(2, 1, 1))
md = md.fit(method_kwargs={'warn_convergence': False})
#prd_trn = best_model.predict(start=0, end=len(train)-1)
#prd_tst = best_model.forecast(steps=len(test))
prd_trn = md.predict(start=0, end=len(train)-1)
prd_tst = md.forecast(steps=len(test))
print(f'\t{measure}={PREDICTION_MEASURES[measure](test, prd_tst)}')

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD1Arima/{file_tag}_p=2_d=1_q=1_arima_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD1Arima/{file_tag}_p=2_d=1_q=1_arima_plots.png', x_label= str(index_multi), y_label=str(target_multi))