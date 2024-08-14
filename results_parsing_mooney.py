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

task = 'Mooney Image'
trials = np.arange(0,24,1)
subjects = df.id

res_correct_u = [] # upright faces
res_correct_i = [] # inverted faces
res_correct_s = [] # scambled faces

for subject in subjects:
    #extract data per subject
    subject_res = df[df['id'] == subject]
    print(f'parsing subject number: {subject}')
          
    stimuli = []
    responses = []
    rts = []
        
    for trial in trials:
        #mooney was the 3rd task
        stimulus = str(3) + '.' + task + '.' + str(trial) + '.' + 'rightAnswer'
        response = str(3) + '.' + task + '.' + str(trial) + '.' + 'answer'
        rt = str(3) + '.' + task + '.' + str(trial) + '.' + 'rt'
        
        stimuli.append(subject_res[stimulus].values[0]) 
        responses.append(subject_res[response].values[0])
        rts.append(subject_res[rt].values[0])
    
    # data validation
    result = all(element == responses[0] for element in responses)
    if result:
        print(f"...repeated entry of: {responses[0]}\n{responses}")            
    
    if float(np.nan) in responses:
        print(f"...some values are missing. Responses are: \n...{responses}")
    
    #print(f'stimuli were: {stimuli}')
    #print(f'responses were: {responses}')
    
    c_correct_u = 0 #upright correct
    c_correct_i = 0 #inverted correct
    c_correct_s = 0 #random/scrambled correct
    
    for s,r in zip(stimuli, responses):
        if s=="U" and s==r:
            c_correct_u+=1
            
        elif s=="I" and r=="U":
            c_correct_i+=1
        
        elif s=="S" and s==r:
            c_correct_s+=1
              
    p_u = c_correct_u / stimuli.count("U") # hit rate
    p_i = c_correct_i / stimuli.count("I") # hit rate
    p_s = 1 - (c_correct_s / stimuli.count("S")) #false alarm rate  = 1 - correct rejection
    
    res_correct_u.append(p_u) # correct upright
    res_correct_i.append(p_i) # correct inverted
    res_correct_s.append(p_s) # false alarm

            
del res_correct_u[15] #excluding one participant with repeated answer
del res_correct_i[15]
del res_correct_s[15]


# for Binomial, sd = sqrt(n*p*(1-p)), where n is # of trials
sem_u = np.sqrt(len(res_correct_u) * np.mean(res_correct_u) * (1-np.mean(res_correct_u)))/(len(res_correct_u))
sem_i = np.sqrt(len(res_correct_i) * np.mean(res_correct_i) * (1-np.mean(res_correct_i)))/(len(res_correct_i))
sem_s = np.sqrt(len(res_correct_s) * np.mean(res_correct_s) * (1-np.mean(res_correct_s)))/(len(res_correct_s))
error = [sem_u*100, sem_i*100, sem_s*100] / np.sqrt(len(res_correct_u))

print(f"upright faces rate: {np.mean(res_correct_u)*100:.1f}% ± {error[0]:.1f}%")
print(f"inverted faces: {np.mean(res_correct_i)*100:.1f}% ± {error[1]:.1f}%")
print(f"random images rate: {np.mean(res_correct_s)*100:.1f}% ± {error[2]:.1f}%")

#visualiztion
plt.rcParams.update({'font.size': 18})
X = ['Random', 'Inverted\nface', 'Upright\nface']
X_axis = np.arange(len(X))
plt.bar(X, [(np.mean(res_correct_s))*100, np.mean(res_correct_i)*100, np.mean(res_correct_u)*100], yerr=error, width=0.5, edgecolor='k', color='None')
plt.axhline(y=50, color='gray', linestyle='--') #chance level

#cosmetics
plt.ylim(0,100)
plt.ylabel('Reported\nface [%]')
plt.xlabel('Stimulus')
sns.despine() #remove top and right box and make square
plt.tight_layout()

plt.show()
#plt.savefig('Mooney',dpi=800)

   
# d' analysis
'''
d' = z(H) - z(F) with loglinear correction. 
With equal number of signal and noise trials:
    - add 0.5 to Hits and FA and 2*0.5 to number of signals and noise trials
With unequal number of trials, as in our case, we need to consider the proportion of trials per condition
'''
n_s = stimuli.count("S")
n_u = stimuli.count("U")
n_i = stimuli.count("I")

#proportion of signal and noise trials
signal_trials_u = n_u / (n_s + n_u)
signal_trials_i = n_i / (n_s + n_i)

noise_trials_u = 1 - signal_trials_u
noise_trials_i = 1 - signal_trials_i

z_p_u = [] #hits upright
z_p_i = [] #hits inverted
z_p_su = [] #false alarms scrambled vs. upright
z_p_si = [] #false alarms scrambled vs. inverted

for r in res_correct_u:
    hits = (r*n_u + signal_trials_u)
    temp_u = norm.ppf(hits / (n_u + 2*signal_trials_u))
    z_p_u.append(temp_u)

for i in res_correct_i:
    hits = i*n_i + signal_trials_i
    temp_i = norm.ppf(hits / (n_i + 2*signal_trials_i))
    z_p_i.append(temp_i)

for s in res_correct_s:
    fa = s*n_s + noise_trials_u
    temp_su = norm.ppf(fa / (n_s + 2*noise_trials_u))
    z_p_su.append(temp_su)
    
    fa = s*n_s + noise_trials_i
    temp_si = norm.ppf(fa / (n_s + 2*noise_trials_i))
    z_p_si.append(temp_si) #false alarms

# d'
u_vs_s = np.subtract(z_p_u, z_p_su)
u_vs_i = np.subtract(z_p_u, z_p_i)
i_vs_s = np.subtract(z_p_i, z_p_si)

dmean_u, err_u = np.mean(u_vs_s), np.std(u_vs_s, ddof=1) / np.sqrt(len(u_vs_s))
dmean_i, err_i = np.mean(u_vs_i), np.std(u_vs_i, ddof=1) / np.sqrt(len(u_vs_i))
dmean_s, err_s = np.mean(i_vs_s), np.std(i_vs_s, ddof=1) / np.sqrt(len(i_vs_s))


print(f'Upright vs scrambled: mean d ± SEM = {dmean_u:.2f} ± {err_u:.2f}')
print(f'Upright vs inverted: mean d ± SEM = {dmean_i:.2f} ± {err_i:.2f}')
print(f'Inverted vs. scrambeled: mean d ± SEM = {dmean_s:.2f} ± {err_s:.2f}')
# Schwedrzik, 2018: 1.78 ± 0.45,
#                   0.94 ± 0.26, 
#                   0.83 ± 0.45 (mean and SD)

# two-sample unpaired t-test, one participant was removed
t ,pval = ttest_ind_from_stats(dmean_u, np.std(u_vs_s, ddof=1), (len(subjects)-1), 
                               1.78, 0.45, 19,
                               equal_var=False,
                               alternative='two-sided')
bf = bayesfactor_ttest(t, (len(subjects)-1), 19, paired=False, alternative='two-sided')
print(f'Upright vs. random:\ntwo-sample t-test: p = {pval:.2f}, t = {t:.2f}, BF = {bf:.2f}')

t2 ,pval2 = ttest_ind_from_stats(dmean_s, np.std(i_vs_s, ddof=1), (len(subjects)-1), 
                               0.83, 0.45, 19, 
                               equal_var=False,
                               alternative='two-sided')
bf2 = bayesfactor_ttest(t2, (len(subjects)-1), 19, paired=False, alternative='two-sided')
print(f'Inverted vs. random:\ntwo-sample t-test: p = {pval2:.2f}, t = {t2:.2f}, BF = {bf2:.2f}')


######################### additional draft code ##########################
#to export data:
#df = pd.DataFrame({'id': np.arange(0,len(u_vs_s)),
#                   'Easy': u_vs_s,
#                   'Medium': u_vs_i,
#                   'Hard': i_vs_s})
#df.to_csv("Mooney_dprime.csv")

#data = np.array([u_vs_s, u_vs_i, i_vs_s])
#print(f'Overall sensativity: mean d = {np.mean(data):.2f} SD {err_s:.2f}')
#ttest_1samp(a = np.mean(data, axis=1), popmean=1.18, alternative='two-sided')

#key-tapping stats
import scipy.stats as st
st.ttest_1samp(a=u_vs_s, popmean=1.78, alternative='two-sided')

right_tapping = [113, 37, 113, 44, 143, 68, 56, 39, 21, 29, 24]
left_tapping = [102, 33, 110, 38, 137, 100, 53, 33, 25, 27, 29]

st.ttest_1samp(a=left_tapping, popmean=63.1, alternative='two-sided')
st.ttest_1samp(a=right_tapping, popmean=59.8, alternative='two-sided')
       


'''
loglin = 0.5

z_p_u = [] #hits upright
z_p_i = [] #hits inverted
z_p_su = [] #false alarms scrambled vs. upright
z_p_si = [] #false alarms scrambled vs. inverted

for r in res_correct_u:
    hits = (r*n_u + loglin)
    temp_u = norm.ppf(hits / (n_u + 2*loglin))
    z_p_u.append(temp_u)

for i in res_correct_i:
    hits = i*n_i + loglin
    temp_i = norm.ppf(hits / (n_i + 2*loglin))
    z_p_i.append(temp_i)

for s in res_correct_s:
    fa = s*n_s + loglin
    temp_su = norm.ppf(fa / (n_s + 2*loglin))
    z_p_su.append(temp_su)
    
    temp_si = norm.ppf(fa / (n_s + 2*loglin))
    z_p_si.append(temp_si) #false alarms
'''