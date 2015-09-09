import numpy as np
import pandas as pd

df = pd.read_csv('shuttle.trn', header=None, index_col=False, sep=' ')

# group by the last column (we will treat groups 2,3,5,6 as outliers)
grouped = df.groupby(9)
outliers = pd.concat([grouped.get_group(2),
                      grouped.get_group(3),
                      grouped.get_group(5),
                      grouped.get_group(6)])
outliers = outliers.drop(9,axis=1)
others = df.drop(grouped.get_group(2).index)
others = others.drop(grouped.get_group(3).index)
others = others.drop(grouped.get_group(5).index)
others = others.drop(grouped.get_group(6).index)
others = others.drop(9,axis=1)

# downsample the outliers to 1.89% of the size of the rest
outliers_sample = outliers.sample(n=int(0.0189*others.shape[0]))

# write out elki file
outliers_sample['class'] = 'Outlier'
others['class'] = 'Normal'
resampled = pd.concat([outliers_sample, others])
desc = resampled[['class']]
resampled = resampled.drop('class', axis=1)

## convert categorical vars 1, 2, 3 to dummy vars
#resampled = resampled.join(pd.get_dummies(resampled[1]))
#resampled = resampled.join(pd.get_dummies(resampled[2]))
#resampled = resampled.join(pd.get_dummies(resampled[3]))
#resampled = resampled.drop(1, axis=1)
#resampled = resampled.drop(2, axis=1)
#resampled = resampled.drop(3, axis=1)

# normalize data
for i in resampled.columns:
    mean = resampled[i].values.mean()
    stddev = resampled[i].values.std()
    if stddev > 0.0:
        resampled[i] = (resampled[i]-mean) / stddev
    else:
        resampled = resampled.drop(i, axis=1)

resampled = resampled.join(desc)

resampled.to_csv('shuttle_for_elki.csv', header=None, index=None)
