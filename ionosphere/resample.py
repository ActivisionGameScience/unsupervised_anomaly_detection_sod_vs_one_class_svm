import numpy as np
import pandas as pd

df = pd.read_csv('ionosphere.data', header=None, index_col=False)

# group by the class column (we will treat the b's as outliers)
# also, drop the class column
grouped = df.groupby(34)
outliers = grouped.get_group('b')
outliers = outliers.drop(34,axis=1)
others = df.drop(grouped.get_group('b').index)
others = others.drop(34,axis=1)

# downsample the b group to 3.4% of the size of the rest
outliers_sample = outliers.sample(n=int(0.034*others.shape[0]))

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

resampled.to_csv('ionosphere_for_elki.csv', header=None, index=None)
