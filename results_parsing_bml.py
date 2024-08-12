# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:42:02 2022

@author: YSK
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from pingouin import bayesfactor_ttest

df = pd.read_csv('Data/data_17.csv')

"""
The format of the columns is:
    taskID.taskName.trial.property
    e.g.,:
    0.Numerosity.0.rightAnswer = stimulus
                .answer = response
                .rt = response time
"""

tasks = ['Numerosity', 'Line Length', 'Biological Motion', 'Mooney image', 'Keyboard Tapping']
trials = np.arange(0,24,1)
subjects = df.id

res_correct_r = []
res_correct_b = []

res_dprime_r = []
res_dprime_b = []

#to extract data per subject
for subject in subjects:
    subject_res = df[df['id'] == subject]
    print(f'...parsing subject number: {subject}')
    for i, task in enumerate(tasks):
        
        stimuli = []
        responses = []
        rts = []
        
        if task == "Biological Motion":            
            for trial in trials:
                stimulus = str(i) + '.' + task + '.' + str(trial) + '.' + 'rightAnswer'
                response = str(i) + '.' + task + '.' + str(trial) + '.' + 'answer'
                rt = str(i) + '.' + task + '.' + str(trial) + '.' + 'rt'
                
                stimuli.append(subject_res[stimulus].values[0])
                if subject_res[response].values[0] == 'None' or subject_res[response].values[0] == "":
                    res = np.nan
                else:
                    res = subject_res[response].values[0]
                responses.append(res)
                rts.append(subject_res[rt].values[0])
                
            
            correct_r = 0
            correct_b = 0
            
            
            for s,r in zip(stimuli, responses):
                
                if s=="Random" and s==r:
                    correct_r+=1
                elif s=="BML" and s==r:
                    correct_b+=1
                    
            p_m = correct_b / stimuli.count("BML")
            p_r = correct_r / stimuli.count("Random")

           
            res_dprime_b.append(st.norm.ppf(st.norm.cdf(p_m)))
            res_dprime_r.append(st.norm.ppf(st.norm.cdf(1-p_r)))
                    
            res_correct_r.append(p_r)
            res_correct_b.append(p_m)
            print(f"--total of {len(stimuli)} trials:\n---random stimuli correct: {correct_r} / {stimuli.count('Random')}\n---bm stimuli correct: {correct_b} / {stimuli.count('BML')}")
   
    
p = np.mean(res_correct_b) # report BM
q = np.mean(res_correct_r) # report random

# for Binomial, se = sqrt(n*p*(1-p))/ sqrt(n), where n is # of trials. 
error = [(np.sqrt(len(res_correct_r)*q*(1-q))*100 / len(res_correct_r)), np.sqrt(len(res_correct_b)*p*(1-p))*100/len(res_correct_b)]

print(f"avg reporting BM ± SEM: {p*100:.1f}% ± {error[1]:.1f}%")
print(f"avg reporting random ± SEM: {(1-q)*100:.1f}% ± {error[0]:.1f}%")

#plot the mean data across participants
X = ['Random', 'Biological']
X_axis = np.arange(len(X))

plt.rcParams.update({'font.size': 18})
plt.clf()
plt.bar(X, [(1-q)*100, p*100], width=0.5, yerr = error, edgecolor='k', color='None')
plt.axhline(y=50, color='gray', linestyle='--') #chance level

plt.ylim(0,100)
plt.ylabel("Reported\nbiological motion [%]")
plt.xlabel("Stimulus")
sns.despine() #remove top and right box and make square
plt.tight_layout()
plt.savefig('MOPP figures/bml',dpi=800)

#d' prime analysis
m_vs_r = np.subtract(res_dprime_b, res_dprime_r)
dmean_m, err_m = np.mean(m_vs_r), np.std(m_vs_r,ddof=1)/np.sqrt(len(m_vs_r))
print(f'Motion: mean {dmean_m:.2f} ± {err_m:.2f}')


# one sample, more reliable given the large sample size of Weil et al., 2018
t1, pval1 = st.ttest_1samp(a=m_vs_r, popmean=0.73, alternative='two-sided')
bf = bayesfactor_ttest(t1, len(subjects), 189, paired=False, alternative='two-sided')
print(f'one-sample t-test: p = {pval1:.2f}, t = {t1:.2f}, BF = {bf:.2f}')

#t2 ,pval2 = st.ttest_ind_from_stats(p*100, np.std(m_vs_r,ddof=1), len(subjects), 0.73, 0.1, nobs2, alternative='two-sided')
#bf = bayesfactor_ttest(1.87, 16, 18, paired=False, alternative='two-sided')
#print("Bayes Factor: %.3f (two-sample unpaired)" % bf)
#np.savetxt("b.csv", res_dprime_b, delimiter=",")
#np.savetxt("r.csv", res_dprime_r, delimiter=",")
'''
#one-tailed tttest to see if BM higher than random
rev_res_correct_r = [1-x for x in res_correct_r]
st.ttest_ind(res_correct_b, rev_res_correct_r, alternative='greater')
'''
        
