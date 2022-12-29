from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots
from ts_functions import HEIGHT, split_dataframe, create_temporal_dataset
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

file_tag = 'glucose_smoothing100'
index_multi = 'Date'
target_multi = 'Glucose'
data_multi = read_csv('../glucose.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)

WIN_SIZE = 100
rolling_multi = data_multi.rolling(window=WIN_SIZE)
smooth_df_multi = rolling_multi.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df_multi[target_multi], title=f'Glucose - Smoothing (win_size={WIN_SIZE})', x_label=index_multi, y_label='glucose level')
plot_series(smooth_df_multi['Insulin'])
xticks(rotation = 45)
show()
savefig(f'imagesD1Transformation/{file_tag}.png')

df = smooth_df_multi.drop('Insulin', axis=1)
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

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD1Transformation/{file_tag}_persistence_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD1Transformation/{file_tag}_persistence_plots.png', x_label=index_multi, y_label=target_multi)

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

plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'imagesD1Transformation/{file_tag}_simpleAvg_eval.png')
plot_forecasting_series(train, test, prd_trn, prd_tst, f'imagesD1Transformation/{file_tag}_simpleAvg_plots.png', x_label=index_multi, y_label=target_multi)