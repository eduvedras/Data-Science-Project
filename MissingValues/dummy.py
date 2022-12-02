#%%
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame, concat,read_csv
from ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder
from numpy import number

register_matplotlib_converters()
file = 'constant'
filename = 'data/diabetic_data_mv_constant.csv'
data = read_csv(filename, na_values='')

def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

variables = get_variable_types(data)
vars = variables['Symbolic'] + variables['Binary']
vars.remove('readmitted')
df = dummify(data, vars)
df = df.drop(['Unnamed: 0'],axis=1)
df.to_csv(f'data/{file}_dummified.csv', index=False)

#Caso nao remova o unamed usa isto abaixo
'''
data = read_csv('data/constant_dummified.csv', na_values='')
data = data.drop(['Unnamed: 0'],axis=1)
data.to_csv(f'data/{file}_dummified1.csv', index=False)'''

# %%
