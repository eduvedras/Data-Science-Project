#%%
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from ds_charts import plot_evaluation_results_tern, bar_chart, plot_evaluation_results
from sklearn.metrics import accuracy_score

file_tag = 'diabetic_sel'
filename = 'dataWeek3/diabetic_sel'
target = 'readmitted'

train: DataFrame = read_csv(f'{filename}_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'{filename}_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

#clf = GaussianNB()
clf = BernoulliNB()
clf.fit(trnX, trnY)

prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results_tern(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'imagesWeek3/{file_tag}_nb_best_bernoulli.png')
show()

estimators = {'GaussianNB': GaussianNB(),
              #'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB(),
              #'CategoricalNB': CategoricalNB
              }

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    yvalues.append(accuracy_score(tstY, prdY))

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'imagesWeek3/{file_tag}_nb_study.png')
show()
# %%
