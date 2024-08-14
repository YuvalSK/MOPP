# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:42:02 2022

@author: YSK
"""
import math
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind_from_stats
from pingouin import bayesfactor_ttest

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

df = pd.read_csv('Data/data_17.csv')

"""
The format of the columns is:
    taskID.taskName.trial.property
    e.g.,:
    0.Numerosity.0.rightAnswer = stimulus
                .answer = response
                .rt = response time
"""

trials = ["Both", "Right", "Left"] # both hands, right and left
task = "Keyboard Tapping"
subjects = df.id
counts = pd.DataFrame()

for trial in range(len(trials)):
    #Key taps was the 4th task
    stimulus = str(4) + '.' + task + '.' + str(trial) + '.' + 'hand'
    response = str(4) + '.' + task + '.' + str(trial) + '.' + 'answer'
    rt = str(4) + '.' + task + '.' + str(trial) + '.' + 'rt'
    
    print(f"Condition: {trials[trial]} hand(s)")
    res = df[response].values
    
    count_sk = []
    for i, r in enumerate(res):
        if isinstance(r, (float, int)) and np.isnan(r):
            #print(f"......missing values")
            count_sk.append(np.nan)
        else:
            count_sk.append(np.char.count(r, sub='sk'))
            print(f"...participant: {i+1}, count: {count_sk[i]}")
    counts[f"{trials[trial]}"] = count_sk

counts.head()
counts.isna().sum()# One participant in the both condition and four in the left hand

err_b = counts['Both'].std(ddof=1)/np.sqrt(len(counts['Both']))
err_r = counts['Right'].std(ddof=1)/np.sqrt(len(counts['Right']))
err_l = counts['Left'].std(ddof=1)/np.sqrt(len(counts['Left']))
error = [err_b, err_r, err_l]
        
print(f"Both: mean ± SEM = {counts['Both'].mean():.2f} ± {err_b:.2f}")
print(f"Right: mean ± SEM = {counts['Right'].mean():.2f} ± {err_r:.2f}")
print(f"Left: mean ± SEM = {counts['Left'].mean():.2f} ± {err_l:.2f}")

#plot the mean data across participants
X = ['Both\nhands', 'Right\nhand', 'Left\nhand']
X_axis = np.arange(len(X))

plt.rcParams.update({'font.size': 18})
plt.clf()
plt.bar(X, [counts['Both'].mean(), counts['Right'].mean(), counts['Left'].mean()], width=0.5, yerr = error, edgecolor='k', color='None')

plt.ylim(0,120)
plt.yticks(np.arange(0, 121, 30.0))

plt.ylabel("# of 'sk' taps [count]")
plt.xlabel("Condition")
sns.despine() #remove top and right box and make square
plt.tight_layout()
plt.show()
#plt.savefig('Taps',dpi=800)

# comparing right and left hands
## For paired t-test, removing empty values in the left condition (no null in right condition)
a = counts[pd.notnull(counts['Left'])]
print(f"(1) comparing right vs. left hand for n = {a['Right'].count()}:")
print(f"...Right: mean ± SEM = {a['Right'].mean():.2f} ± {a['Right'].std(ddof=1)/np.sqrt(len(a['Right'])):.2f}")
print(f"...Left: mean ± SEM = {a['Left'].mean():.2f} ± {a['Left'].std(ddof=1)/np.sqrt(len(a['Left'])):.2f}")

t ,pval = ttest_rel(a['Right'], a['Left'], alternative='two-sided')
bf = bayesfactor_ttest(t, a["Right"].count(), paired=True, alternative='two-sided')
print(f'......Paired t-test left vs. right: p = {pval:.2f}, t = {t:.2f}, BF = {bf:.2f}')

# pooling over a single hand analysis
pooled_mean = counts[['Right','Left']].stack().mean()
sd_mean = counts[['Right','Left']].stack().std(ddof=1)
count_mean = counts[['Right','Left']].stack().count()
print(f"(2) pooling over a single hand for n = {count_mean}: mean ± SEM = {pooled_mean:.2f} ± {sd_mean/np.sqrt(count_mean):.2f}")

## Noyce 2014 reported mean over a single hand 60.3 and CI = [57.6, 63.0]
## calculating SEM:
n = 93
ci_upper = 63.0
pop_mean = 60.3
sd = (ci_upper - pop_mean)  * np.sqrt(n) / 1.96
sem =  sd / np.sqrt(n)
print(f"...Noyce et al., 2014 tested N = {n}\n...mean ± sem: {pop_mean:.2f} ± {sem:.2f}")

## two-sample unpaired t-test, one participant was removed
t2 ,pval2 = ttest_ind_from_stats(pooled_mean, sd_mean, count_mean, 
                               pop_mean, sd, n,
                               equal_var=False,
                               alternative='two-sided')
bf2 = bayesfactor_ttest(t2, count_mean, n, paired=False, alternative='two-sided')
print(f'......Pooled single hand:\ntwo-sample t-test: p = {pval2:.2f}, t = {t2:.2f}, BF = {bf2:.2f}')
      


