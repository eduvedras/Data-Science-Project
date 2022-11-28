#%%
from pandas import read_csv, Series
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots, Axes
from ds_charts import bar_chart, get_variable_types, choose_grid, HEIGHT, multiple_bar_chart, multiple_line_chart
from seaborn import distplot
from scipy.stats import norm, expon, lognorm
from numpy import log
import numpy as np
import random
import pandas as pd


register_matplotlib_converters()
filename = '../drought.csv'
#filename = '../diabetic_data.csv'
#data = read_csv(filename, index_col='encounter_id', na_values='')
data = read_csv(filename, parse_dates=['date'], infer_datetime_format=True)
#pandas dataframe





#Salva o array como uma lista e ? cria um dicionario
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()

    # Gaussian
    mean, sigma = norm.fit(x_values)
    # MÃ©dia e desvio padrÃ£o da amostra (numpy.float64)

    x_values = x_values.tolist()

    random.shuffle(x_values)

    x_values = x_values[0:1000]

    x_values.sort()

    #x_values = series.sort_values().values


    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)



    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    #LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)



    return distributions, x_values



#Recebe a posiÃ§Ã£o (i, j), os dados da variÃ¡vel salva em "data", e o nome da variÃ¡vel
def histogram_with_distributions(ax: Axes, series: Series, var: str):

   
    values = series.sort_values().values
    # Array do numpy
    
    ax.hist(values, 20, density=True)
    #plota o histograma com base nos valores recebidos. Adota bins=20

    #Chama a funÃ§Ã£o "compute_known_distributions" enviando o array no numpy (valores)
    distributions, values = compute_known_distributions(values)


    #Adiciona a distribuiÃ§Ã£o
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')
    
    print("1 more done")



#distribution for numeric



numeric_vars = get_variable_types(data)['Numeric']
#list


if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')


rows, cols = choose_grid(len(numeric_vars))
#escolhe a quantidade de linhas e colunas para os grÃ¡ficos

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
#parÃ¢metros do matplotlib

i, j = 0, 0 #Contadores
for n in range(len(numeric_vars)):

    #Chama a funÃ§Ã£o "histogram_with_distributions"
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    #Envia a posiÃ§Ã£o (i, j), os dados da variÃ¡vel salva em "data", e o nome da variÃ¡vel

    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1) #Altera o contador

savefig('imageD2/histogram_numeric_distribution1000.png') # salva
show()
# %%
