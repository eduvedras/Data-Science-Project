from torch import zeros
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.autograd import Variable
from pandas import read_csv, Series

class DS_LSTM(Module):
    def __init__(self, input_size, hidden_size, learning_rate, num_layers=1, num_classes=1):
        super(DS_LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = Linear(hidden_size, self.num_classes)
        self.criterion = MSELoss()    # mean-squared error for regression
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        h_0 = Variable(zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(zeros(
            self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

    def fit(self, trainX, trainY):
        # Train the model
        outputs = self(trainX)
        self.optimizer.zero_grad()
        # obtain the loss function
        loss = self.criterion(outputs, trainY)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, data):
        # Predict the target variable for the input data
        return self(data).detach().numpy()



from pandas import read_csv, DataFrame, Series
from torch import manual_seed, Tensor
from torch.autograd import Variable
from ts_functions import split_dataframe, sliding_window
from sklearn.preprocessing import MinMaxScaler
from ds_charts import HEIGHT, multiple_line_chart
from matplotlib.pyplot import subplots, show, savefig
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, sliding_window

target = 'QV2M'
index_col='date'

file_tag = 'drought_lstm'
data = read_csv('../droughtDrop.csv', index_col=index_col, sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
nr_features = len(data.columns)

def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

agg_multi_df = aggregate_by(data, index_col, 'M')

WIN_SIZE = 50
rolling_multi = agg_multi_df.rolling(window=WIN_SIZE)
smooth_df = rolling_multi.mean()
smooth_df.drop(index=smooth_df.index[:WIN_SIZE], axis=0, inplace=True)
sc = MinMaxScaler()
data = DataFrame(sc.fit_transform(smooth_df), index=smooth_df.index, columns=smooth_df.columns)
manual_seed(1)
train, test = split_dataframe(data, trn_pct=.70)

best = ('',  0, 0.0)
last_best = -100
best_model = None

measure = 'R2'
flag_pct = False

learning_rate = 0.001
sequence_size = [4, 20, 60, 100]
nr_hidden_units = [8, 16, 32]
max_iter = [500, 500, 1500, 2500]
episode_values = [max_iter[0]]
for el in max_iter[1:]:
    episode_values.append(episode_values[-1]+el)

nCols = len(sequence_size)
_, axs = subplots(1, nCols, figsize=(nCols*HEIGHT, HEIGHT), squeeze=False)
values = {}
for s in range(len(sequence_size)):
    length = sequence_size[s]
    trnX, trnY = sliding_window(train, seq_length = length)
    trnX, trnY  = Variable(Tensor(trnX)), Variable(Tensor(trnY))
    tstX, tstY = sliding_window(test, seq_length = length)
    tstX, tstY  = Variable(Tensor(tstX)), Variable(Tensor(tstY))

    for k in range(len(nr_hidden_units)):
        hidden_units = nr_hidden_units[k]
        yvalues = []
        model = DS_LSTM(input_size=nr_features, hidden_size=hidden_units, learning_rate=learning_rate)
        next_episode_i = 0
        for n in range(1, episode_values[-1]+1):
            model.fit(trnX, trnY)
            if n == episode_values[next_episode_i]:
                next_episode_i += 1
                prd_tst = model.predict(tstX)
                yvalues.append((PREDICTION_MEASURES[measure])(tstY, prd_tst))
                print((f'LSTM - seq length={length} hidden_units={hidden_units} and nr_episodes={n}->{yvalues[-1]:.2f}'))
                if yvalues[-1] > last_best:
                    best = (length, hidden_units, n)
                    last_best = yvalues[-1]
                    best_model = model
        values[hidden_units] = yvalues

    multiple_line_chart(
        episode_values, values, ax=axs[0, s], title=f'LSTM seq length={length}', xlabel='nr episodes', ylabel=measure, percentage=flag_pct)
print(f'Best results with seq length={best[0]} hidden={best[1]} episodes={best[2]} ==> measure={last_best:.2f}')
savefig(f'imagesD2ModelEval/{file_tag}_lstm_study.png')
show()