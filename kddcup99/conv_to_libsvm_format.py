import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

df = pd.read_csv('kddcup_for_elki_5000.csv', header=None, index_col=False)
labelix = df.shape[1]-1

labels = df[labelix]
df = df.drop(labelix, axis=1)

f = open('kddcup_for_libsvm_5000.txt', 'w')

for i in range(0,df.shape[0]):
    f.write('+1 ')
    for j in range(0,df.shape[1]):
        f.write('%d:%f ' % (j+1, df.iloc[i][j]))
    f.write('\n')
