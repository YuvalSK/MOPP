# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:42:02 2022

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, ttest_ind, ttest_1samp


df = pd.read_csv('Data/data_17.csv')

tasks = ['Numerosity', 'Line Length', 'Biological Motion', 'Mooney Image', 'Keyboard Tapping']
trials = np.arange(0,24,1)
subjects = df.id

res_correct_u = [] # upright faces
res_correct_i = [] # inverted faces
res_correct_s = [] # scambled faces

for subject in subjects:
    #extract data per subject
    subject_res = df[df['id'] == subject]
    print(f'--parsing subject number: {subject}')
    for i, task in enumerate(tasks):
        
        stimuli = []
        responses = []
        rts = []
        
        if task == "Mooney Image":     
            for trial in trials:
                stimulus = str(i) + '.' + task + '.' + str(trial) + '.' + 'rightAnswer'
                response = str(i) + '.' + task + '.' + str(trial) + '.' + 'answer'
                rt = str(i) + '.' + task + '.' + str(trial) + '.' + 'rt'
                
                stimuli.append(subject_res[stimulus].values[0])
                s_response = subject_res[response].values[0]
                if s_response == 'None' or s_response == "" :
                    print(f"---missing values for participant: {subject}")
                    continue
                responses.append(s_response)
                rts.append(subject_res[rt].values[0])
                
            result = all(element == responses[0] for element in responses)
            if result:
                print(f"----repeated entry\n{responses[0]}")            
                continue
            
            c_correct_u = 0
            c_correct_i = 0
            c_correct_s = 0
            
            #print(f'stimuli were: {stimuli}')
            #print(f'responses were: {responses}')
            
            for s,r in zip(stimuli, responses):
                if s=="U" and s==r:
                    c_correct_u+=1
                    
                elif s=="I" and r=="U":
                    c_correct_i+=1
                
                elif s=="S" and s==r:
                    c_correct_s+=1
            
                  
            p_u = c_correct_u / stimuli.count("U") #hits
            p_i = c_correct_i / stimuli.count("I") #hits
            p_s = 1 - (c_correct_s / stimuli.count("S")) #1 - correct rejection = false alarms 
            
            res_correct_u.append(p_u) # correct upright
            res_correct_i.append(p_i) # correct inverted
            res_correct_s.append(p_s) # false alarms
            
        
X = ['Random', 'Inverted\nface', 'Upright\nface']
X_axis = np.arange(len(X))

from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
plt.rcParams.update({'font.size': 18})
plt.clf()

'''
group data
'''
# for Binomial, sd = sqrt(n*p*(1-p)), where n is # of trials
sem_u = np.sqrt(len(res_correct_u) * np.mean(res_correct_u) * (1-np.mean(res_correct_u)))/(len(res_correct_u))
sem_i = np.sqrt(len(res_correct_i) * np.mean(res_correct_i) * (1-np.mean(res_correct_i)))/(len(res_correct_i))
sem_s = np.sqrt(len(res_correct_s) * np.mean(res_correct_s) * (1-np.mean(res_correct_s)))/(len(res_correct_s))
error = [sem_u*100, sem_i*100, sem_s*100] / np.sqrt(len(res_correct_u))

print(f"discrimination of upright faces: {np.mean(res_correct_u)*100:.1f}% ± {error[0]:.1f}%")
print(f"discrimination of inverted faces: {np.mean(res_correct_i)*100:.1f}% ± {error[1]:.1f}%")
print(f"discrimination of random images: {np.mean(res_correct_s)*100:.1f}% ± {error[2]:.1f}%")


plt.rcParams.update({'font.size': 18})
plt.bar(X, [(np.mean(res_correct_s))*100, np.mean(res_correct_i)*100, np.mean(res_correct_u)*100], yerr=error, width=0.5, edgecolor='k', color='None')
plt.ylim(0,100)

plt.ylabel('Reported\nface [%]')
plt.xlabel('Stimulus')


#remove top and right box and make square
sns.despine()

#change level
plt.axhline(y=50, color='gray', linestyle='--')
plt.show()
plt.tight_layout()

#plt.savefig('MOPP figures/Mooney',dpi=800)

   
# for d' = z(H) - z(F) with loglinear correction. 
# For loglinear method with equal number of signal and noise trials add 0.5 to hits and FA and 2*0.5 to nubmer of signals and noise trials
# here the number of trials is not equal so I need to take into account the proportion of trials per condition

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

#number of trials out of 24:
n_s = 11
n_u = 5
n_i = 8

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
# Schwedrzik, 2018: 1.78 ± 0.45, 0.94 ± 0.26, and 0.83 ± 0.45 (SDs)

df = pd.DataFrame({'id': np.arange(0,len(u_vs_s)),
                   'Easy': u_vs_s,
                   'Medium': u_vs_i,
                   'Hard': i_vs_s})
df.to_csv("Mooney_dprime.csv")

#data = np.array([u_vs_s, u_vs_i, i_vs_s])
#print(f'Overall sensativity: mean d = {np.mean(data):.2f} SD {err_s:.2f}')
#ttest_1samp(a = np.mean(data, axis=1), popmean=1.18, alternative='two-sided')


'''
individual data
'''
# for Binomial, sd = sqrt(n*p*(1-p)), where n is # of trials
sem_u = np.sqrt(len(res_correct_u) * np.mean(res_correct_u) * (1-np.mean(res_correct_u)))/(len(res_correct_u))
sem_i = np.sqrt(len(res_correct_i) * np.mean(res_correct_i) * (1-np.mean(res_correct_i)))/(len(res_correct_i))
sem_s = np.sqrt(len(res_correct_s) * np.mean(res_correct_s) * (1-np.mean(res_correct_s)))/(len(res_correct_s))
error = [sem_u*100, sem_i*100, sem_s*100] / np.sqrt(len(res_correct_u))

print(np.mean(res_correct_u))
print(np.mean(res_correct_i))
print(np.mean(res_correct_s))


plt.rcParams.update({'font.size': 18})
plt.bar(X, [(1-np.mean(res_correct_s))*100, np.mean(res_correct_i)*100, np.mean(res_correct_u)*100], yerr=error, width=0.5, edgecolor='k', color='None')
plt.ylim(0,100)

plt.ylabel('Reported\nface [%]')
plt.xlabel('Stimulus')


#remove top and right box and make square
sns.despine()

#change level
plt.axhline(y=50, color='gray', linestyle='--')

plt.show()


plt.tight_layout()
plt.savefig('MOPP figures/Mooney',dpi=800)



#key-tapping results
import scipy.stats as st
st.ttest_1samp(a=u_vs_s, popmean=1.78, alternative='two-sided')

right_tapping = [113, 37, 113, 44, 143, 68, 56, 39, 21, 29, 24]
left_tapping = [102, 33, 110, 38, 137, 100, 53, 33, 25, 27, 29]

st.ttest_1samp(a=left_tapping, popmean=63.1, alternative='two-sided')
st.ttest_1samp(a=right_tapping, popmean=59.8, alternative='two-sided')
       
"""
The format of the columns is:
    taskID.taskName.trial.property
    e.g.,:
    0.Numerosity.0.rightAnswer = stimulus
                .answer = response
                .rt = response time
"""