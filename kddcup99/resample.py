import numpy as np
import pandas as pd

df = pd.read_csv('kddcup.data_10_percent', header=None, index_col=False)

# group by the last column
grouped = df.groupby(41)
others = grouped.get_group('normal.')
outliers = df.drop(grouped.get_group('normal.').index)

# let's subsample
others = others.sample(n=10000)

## downsample the outliers to 2% of the size of the rest
outliers_sample = outliers.sample(n=int(0.02*others.shape[0]))

outliers_sample['class'] = 'Outlier'
others['class'] = 'Normal'
resampled = pd.concat([outliers_sample, others])
desc = resampled[['class']]
resampled = resampled.drop(41, axis=1)
resampled = resampled.drop('class', axis=1)

# convert categorical vars 1, 2, 3 to dummy vars
resampled = resampled.join(pd.get_dummies(resampled[1]))
resampled = resampled.join(pd.get_dummies(resampled[2]))
resampled = resampled.join(pd.get_dummies(resampled[3]))
resampled = resampled.drop(1, axis=1)
resampled = resampled.drop(2, axis=1)
resampled = resampled.drop(3, axis=1)

# normalize data
for i in resampled.columns:
    mean = resampled[i].values.mean()
    stddev = resampled[i].values.std()
    if stddev > 0.0:
        resampled[i] = (resampled[i]-mean) / stddev
    else:
        resampled = resampled.drop(i, axis=1)

resampled = resampled.join(desc)

### write out elki file
resampled.to_csv('kddcup_for_elki_10000.csv', header=None, index=None)
