# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:42:02 2022

@author: YSK
"""
import math
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind, ttest_1samp, ttest_ind_from_stats
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
res = []

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
            print(f"......trial: {i} is missing values")
            count_sk.append(np.nan)
        else:
            count_sk.append(np.char.count(r, sub='sk'))
            print(f"...participant: {i+1}, count: {count_sk[i]}")
    counts[f"{trials[trial]}"] = count_sk

    
result = responses.count("sk")
    

            
del res_correct_u[15] #excluding one participant with repeated answer


# for Binomial, sd = sqrt(n*p*(1-p)), where n is # of trials
#key-tapping stats

right_tapping = [113, 37, 113, 44, 143, 68, 56, 39, 21, 29, 24]
left_tapping = [102, 33, 110, 38, 137, 100, 53, 33, 25, 27, 29]

st.ttest_1samp(a=left_tapping, popmean=63.1, alternative='two-sided')
st.ttest_1samp(a=right_tapping, popmean=59.8, alternative='two-sided')
       


