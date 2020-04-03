import numpy as np
import pandas as pd
dataset=pd.read_csv('interventions.csv')
temp=pd.read_csv('interventions.csv')
print(dataset.isnull().sum().sum())
x=dataset.iloc[:,3:].values
y=dataset.iloc[:,3:].values
#Deletion if column has null in it
a=['stay at home','>50 gatherings','>500 gatherings','public schools','restaurant dine-in','entertainment/gym','federal guidelines','foreign travel ban']
for i in range(0,8):
    n=temp[a[i]]
    t=n.isnull().sum().sum()
    if(t!=0):
        temp.drop(columns=a[i],inplace=True)
#Replacing with mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy= 'mean',verbose=0)
imputer = imputer.fit(x[:,:])
x[:,:] = imputer.transform(x[:,:])