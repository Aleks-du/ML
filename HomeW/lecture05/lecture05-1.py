#Дан файл "AnuranCalls(MFCCs).zip", в котором содержится обработка звуков издаваемых различными видами лягушек. В качестве признаков выступают мел-кепстральные коэффициенты MFCC
#1. Нормализовать данные
%matplotlib inline
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

z = zipfile.ZipFile("Anuran_Calls_(MFCCs).zip")
df = pd.read_csv(z.open("Frogs_MFCCs.csv"))

#don't work
#new_df = df.loc[['MFCCs_ 1' : 'MFCCs_22']] 
new_df = df[['MFCCs_ 1','MFCCs_ 2','MFCCs_ 3','MFCCs_ 4','MFCCs_ 5','MFCCs_ 6','MFCCs_ 7','MFCCs_ 8','MFCCs_ 9','MFCCs_10','MFCCs_11','MFCCs_12','MFCCs_13','MFCCs_14','MFCCs_15','MFCCs_16','MFCCs_17','MFCCs_18','MFCCs_19','MFCCs_20','MFCCs_21','MFCCs_22',]]
scaler = StandardScaler()
scaler.fit(new_df)
new_df = scaler.fit_transform(new_df)

res = pd.DataFrame(new_df)
res.head()
