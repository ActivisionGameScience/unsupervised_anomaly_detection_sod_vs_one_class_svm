import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

df = pd.read_csv('outlier_scores.csv', header=None, index_col=False)

# group by the first column
df['binary_truth'] = 0
grouped = df.groupby(0)
outliers = grouped.get_group('Outlier')
df['binary_truth'].iloc[outliers.index] = 1

print("ROC-AUC score %f" % roc_auc_score(df['binary_truth'].values, df[1].values))

roc_curve_fpr, roc_curve_tpr, roc_curve_thresholds = roc_curve(df['binary_truth'].values, df[1].values)
for i in range(0, roc_curve_fpr.shape[0]):
    print("%f, %f" % (roc_curve_fpr[i], roc_curve_tpr[i]))
