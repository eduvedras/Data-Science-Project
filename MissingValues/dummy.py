#%%
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame, concat,read_csv
from ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder
from numpy import number

register_matplotlib_converters()
file = 'frequent'
filename = 'data/diabetic_data_mv_most_frequent.csv'
data = read_csv(filename, na_values='')
specialty_values = []
speenc_values = []

def specialty_encoding(x):
    if x not in specialty_values:
        specialty_values.append(x)
        
def diag(x):
    if 'Missing' in x:
        x = '0'
    if 'V' in x:
        x = x.replace("V", "-2.")
    if 'E' in x:
        x = x.replace("E","-1.")
    return x

def aux(x):
    a = float(x)


data['race'] = data['race'].replace(['Caucasian', 'Asian', 'AfricanAmerican', 'Hispanic', 'Other','Missing'], ['1','2','3','4','5','0'])
data['gender'] = data['gender'].replace(['Male','Female','Unknown/Invalid'], ['1','2','0'])
data['age'] = data['age'].replace(['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)','[50-60)','[60-70)','[70-80)','[80-90)','[90-100)'], ['1','2','3','4','5','6','7','8','9','10'])

data['medical_specialty'].apply(specialty_encoding)
i=0
speenc_values.append(str(i+1))
while i < len(specialty_values):
    i = i + 1
    if i == 1:
        speenc_values.append(str(i-1))
        continue
    if(i != len(specialty_values)):
        speenc_values.append(str(i))

data['medical_specialty'] = data['medical_specialty'].replace(specialty_values, speenc_values)
data['diag_1'] = data['diag_1'].apply(diag)
data['diag_2'] = data['diag_2'].apply(diag)
data['diag_3'] = data['diag_3'].apply(diag)
data['max_glu_serum'] = data['max_glu_serum'].replace(['>200','>300','Norm','None'],['2','3','1','0'])
data['A1Cresult'] = data['A1Cresult'].replace(['None','Norm','>7','>8'],['0','1','2','3'])
data['metformin'] = data['metformin'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['repaglinide'] = data['repaglinide'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['nateglinide'] = data['nateglinide'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['chlorpropamide'] = data['chlorpropamide'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['glimepiride'] = data['glimepiride'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['glipizide'] = data['glipizide'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['glyburide'] = data['glyburide'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['pioglitazone'] = data['pioglitazone'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['rosiglitazone'] = data['rosiglitazone'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['acarbose'] = data['acarbose'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['miglitol'] = data['miglitol'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['tolazamide'] = data['tolazamide'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['examide'] = data['examide'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['citoglipton'] = data['citoglipton'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['insulin'] = data['insulin'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['glyburide-metformin'] = data['glyburide-metformin'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['acetohexamide'] = data['acetohexamide'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['tolbutamide'] = data['tolbutamide'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['troglitazone'] = data['troglitazone'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['glipizide-metformin'] = data['glipizide-metformin'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['glimepiride-pioglitazone'] = data['glimepiride-pioglitazone'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['metformin-rosiglitazone'] = data['metformin-rosiglitazone'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['metformin-pioglitazone'] = data['metformin-pioglitazone'].replace(['No','Down','Steady','Up'],['0','1','2','3'])
data['change'] = data['change'].replace(['No','Ch'],['0','1'])
data['diabetesMed'] = data['diabetesMed'].replace(['No','Yes'],['0','1'])
#data['readmitted'] = data['readmitted'].replace(['<30','>30','NO'],['1','2','0'])
#data.apply(aux)
#print(data.to_string())

data = data.drop(['Unnamed: 0'],axis=1)
data.to_csv(f'data/{file}_dummified.csv', index=False)

#Caso nao remova o unamed usa isto abaixo
'''
data = read_csv('data/constant_dummified.csv', na_values='')
data = data.drop(['Unnamed: 0'],axis=1)
data.to_csv(f'data/{file}_dummified1.csv', index=False)'''

# %%
