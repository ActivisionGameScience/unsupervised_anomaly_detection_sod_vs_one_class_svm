import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

df = pd.read_csv('kddcup_for_elki_100000.csv', header=None, index_col=False)
labelix = df.shape[1]-1

labels = df[labelix]
df = df.drop(labelix, axis=1)

svm = OneClassSVM(kernel='rbf', gamma=1.0/df.shape[0], tol=0.001, nu=0.5, shrinking=True, cache_size=80)
svm = svm.fit(df.values)

scores = svm.decision_function(df.values).flatten()
maxvalue = np.max(scores)
scores = maxvalue - scores

output = pd.DataFrame()

# perform reverse sort
sort_ix = np.argsort(scores)[::-1]

output['labels'] =  labels[sort_ix]
output['outlier_scores'] =  scores[sort_ix]

output.to_csv('outlier_scores.csv', header=None, index=None)
