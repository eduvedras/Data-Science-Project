from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, subplots
from ts_functions import HEIGHT, split_dataframe, create_temporal_dataset
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series

file_tag = 'glucose_diff_2'
index_multi = 'Date'
target_multi = 'Glucose'
data_multi = read_csv('../glucose.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)

diff_df_multi = data_multi.diff()
diff_df_multi = diff_df_multi.diff()
#diff_df_multi = data_multi
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df_multi[target_multi], title='Glucose - Differentiation', x_label=index_multi, y_label='glucose level')
xticks(rotation = 45)
show()
savefig(f'imagesD1Transformation/{file_tag}.png')

df = diff_df_multi
df.drop(index=df.index[:2], axis=0, inplace=True)
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