# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:42:02 2022
@author: YSK
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import math
import scipy
import scipy.stats as st
import statsmodels.api as sm
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

tasks = ['Numerosity', 'Line Length']
trials = np.arange(0,24,1)
subjects = df.id

df_res_n = pd.DataFrame()
df_res_l = pd.DataFrame()
n_stimuli = []

for subject in subjects:
    subject_res = df[df['id'] == subject]
    print(f'parsing subject number: {subject}')
    
    for i, task in enumerate(tasks):
        print(f"...task: {task}")
        stimuli = []
        responses = []
        rts = []
        
        if task == 'Numerosity' or task == 'Line Length':
            for trial in trials:
                stimulusCol = str(i) + '.' + task + '.' + str(trial) + '.' + 'rightAnswer'
                responseCol = str(i) + '.' + task + '.' + str(trial) + '.' + 'answer'
                rtCol = str(i) + '.' + task + '.' + str(trial) + '.' + 'rt'
                stimuli.append(subject_res[stimulusCol].values[0])
                if subject_res[responseCol].values[0] == 'None' or math.isnan(subject_res[responseCol].values[0]) or subject_res[responseCol].values[0] == " ":
                    res = np.nan
                    print(f"......missing response in trial: {trial+1}")
                else:
                    res = int(subject_res[responseCol].values[0])
                responses.append(res)
                rts.append(subject_res[rtCol].values[0])
            
            if task == 'Numerosity':
                n_stimuli = stimuli
                r_title = 'N-x' + str(subject)
                df_res_n[r_title] = responses  
                
            if task == "Line Length":
                l_stimuli = stimuli
                r_title = 'L-x' + str(subject)
                df_res_l[r_title] = responses 
        

# (1) analysing of the length task                
del df_res_l["L-x10"] # subject 10 was missing three trials
del df_res_l["L-x16"] # subject 16 was missing four trials


df_res_l['s'] = l_stimuli # same stimulus was used
df_res_l['x_mean'] = df_res_l.iloc[:, 0:-1].mean(axis=1)
df_res_l['x_SD'] = df_res_l.iloc[:, 0:-1].std(axis=1)

df_res_l.duplicated(subset=['s']).value_counts() # ten magnitudes are duplicated
# we pool over them to calculate the mean per stimuli across participants
df_res_l_without_dup = df_res_l.groupby('s').mean().reset_index()

# visualization
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
plt.rcParams.update({'font.size': 18})
plt.clf()

# calculate se errors. Two subjects were excluded
e1 = df_res_l_without_dup['x_SD']/np.sqrt(len(subjects)-2) 
plt.errorbar(df_res_l_without_dup['s'], df_res_l_without_dup['x_mean'], yerr=e1, fmt='o', c='k', label='Length')

# linear regression
x , y = df_res_l_without_dup['s'], df_res_l_without_dup['x_mean']
m, b = np.polyfit(x, y, 1)
slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
print(f'Length\nR square linear: {r_value**2:.2f}, \np value: {p_value} ')

plt.plot(x, m*x+b, c='k')

lim = 20
plt.plot([0,lim], [0,lim], c='gray', linestyle='--') #unity line

# cosmetics
plt.xlabel('Stimulus [A.U.]')
plt.ylabel('Estimate [A.U.]')
sns.despine() # remove top and right box and make square
plt.xlim([0,lim])
plt.xticks(np.arange(0, 21, 5.0))
plt.ylim([0,lim])
plt.axis('scaled')
plt.tight_layout()
plt.show()
#plt.savefig('Length.png',dpi=800)


# (2) analysing the numerosity task

df_res_n['s'] = n_stimuli #same stimulus was used
df_res_n['x_mean'] = df_res_n.iloc[:, 0:-1].mean(axis=1)
df_res_n['x_SD'] = df_res_n.iloc[:, 0:-1].std(axis=1)

df_res_n.duplicated(subset=['s']).value_counts() #eight magnitudes are duplicated
# we pool over them to calculate the mean per stimuli across participants
df_res_n_without_dup = df_res_n.groupby('s').mean().reset_index()

# linear regression between response and log of the stimuli
## as in previous studies, only magnitudes exceding 10 are included 
df_res_n_without_dup_10 = df_res_n_without_dup[df_res_n_without_dup.s > 10]
x , y = df_res_n_without_dup_10['s'], df_res_n_without_dup_10['x_mean']
t = np.log(x)
p = np.polyfit(t, y, 1)
slope, intercept, r_value, p_value, std_err = st.linregress(t, y)
print(f'Numerosity\nR square log: {r_value**2:.2f} \np value: {p_value}')

a = p[0] #intercept
b = p[1] #slope
x_fitted = np.linspace(np.min(x), np.max(x), 100)
y_fitted = a * np.log(x_fitted) + b

#visualization
plt.clf()
e2 = df_res_n_without_dup['x_SD']/np.sqrt(len(subjects))
plt.errorbar(df_res_n_without_dup['s'], df_res_n_without_dup['x_mean'], yerr=e2, fmt='o', c='k', label='Numerosity')
plt.plot(x_fitted, y_fitted, c='k')

#cosmetics
lim = 40
plt.xlim([0,lim])
plt.ylim([0,lim])
plt.xlabel('Stimulus [dots]') #fit with log stimuli
plt.ylabel('Estimate [dots]')
plt.plot([0,lim], [0,lim], c='gray', linestyle='--') #unity line
sns.despine() #remove top and right box and make square
plt.axis('square')
plt.xticks(np.arange(0, 41, 10.0))
plt.tight_layout()

plt.show()
#plt.savefig('Numerosity.png',dpi=800)
    
        
