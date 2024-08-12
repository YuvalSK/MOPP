# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:42:02 2022
need to cintinue comparison with LMM and errors
@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import statsmodels.api as sm
import seaborn as sns

df = pd.read_csv('Data/data_17.csv')

tasks = ['Numerosity', 'Line Length']
trials = np.arange(0,24,1)
subjects = df.id

df_res_n = pd.DataFrame()
df_res_l = pd.DataFrame()
n_stimuli = []

for subject in subjects:
    #extract data per subject
    subject_res = df[df['id'] == subject]
    print(f'parsing subject number: {subject}')
    
    for i, task in enumerate(tasks):
        print(f"-{task}")
        
        stimuli = []
        responses = []
        rts = []
        
        if task == 'Numerosity' or task == 'Line Length':
            for trial in trials:
                stimulusCol = str(i) + '.' + task + '.' + str(trial) + '.' + 'rightAnswer'
                responseCol = str(i) + '.' + task + '.' + str(trial) + '.' + 'answer'
                rtCol = str(i) + '.' + task + '.' + str(trial) + '.' + 'rt'
                stimuli.append(subject_res[stimulusCol].values[0])
                print(subject_res[responseCol].values[0])
                if subject_res[responseCol].values[0] == 'None' or math.isnan(subject_res[responseCol].values[0]) or subject_res[responseCol].values[0] == " ":
                    res = -1
                else:
                    res = int(subject_res[responseCol].values[0])
                responses.append(res)
                rts.append(subject_res[rtCol].values[0])
                
            '''
            #plot response times by trials
            plt.scatter(trials, rts, edgecolor='k', facecolor = 'none')
            
            rts = [0.0001 if math.isnan(r) else r for r in rts]
            z = np.polyfit(trials, rts, 1)
            p = np.poly1d(z)
            
            lim = np.max(rts) + 1000
            plt.xlim([-1,len(trials)+1])
            plt.ylim([0,lim])
            plt.plot(trials, p(trials), c='r', linestyle="-")
            plt.title(f'{task} task - subject: {subject}')
            plt.grid()
            plt.xlabel('Trial [#]')
            plt.ylabel('Response time [msec]')
            plt.savefig(f"MOPP figures/results/{task}_s{subject}_rt")
            
            #plot response by stimulus
            plt.clf()
            plt.scatter(stimuli, responses, edgecolor='k', facecolor= 'none')
            #lim = np.max(responses) + 10
            lim = 30
            plt.plot([0,lim], [0,lim],c='k')
            plt.title(f'{task} task - subject: {subject}')
            plt.xlim([0,lim])
            plt.ylim([0,lim])
            plt.xlabel('Stimulus [units]')
            plt.ylabel('Response [units]')
            plt.savefig(f"MOPP figures/results/{task}_s{subject}")
            '''
            
            if task == 'Numerosity':
                n_stimuli = stimuli
                r_title = 'N-x' + str(subject)
                df_res_n[r_title] = responses  
                
            if task == "Line Length":
                l_stimuli = stimuli
                r_title = 'L-x' + str(subject)
                df_res_l[r_title] = responses 
        
                
#length, fitting linear regression    
del df_res_l["L-x10"] # missing three trials
del df_res_l["L-x16"] # missing four trials


df_res_l['s'] = l_stimuli #same stimulus was used
df_res_l['x_mean'] = df_res_l.iloc[:, 0:-1].mean(axis=1)
df_res_l['x_SD'] = df_res_l.iloc[:, 0:-1].std(axis=1)


print(df_res_l.duplicated())
df_res_l_without_dup = df_res_l.groupby('s').mean().reset_index()

#visualization
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
plt.rcParams.update({'font.size': 18})
plt.clf()

#time domain analysis
''' 
plt.clf()
#plt.errorbar([i for i in range(len(e2))], (df_res_l['x_mean'] - df_res_l['s'])/df_res_l['x_mean'], yerr=e2, fmt='o', c='k', label='Length')

plt.scatter([i for i in range(len(e2))], (df_res_l['L-x1'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x2'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x3'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x4'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x5'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x6'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x7'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x8'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x9'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x10'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x11'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x12'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x13'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x14'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x15'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x16'] - df_res_l['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_l['L-x17'] - df_res_l['s']), facecolors='none', edgecolor='gray')

plt.scatter([i for i in range(len(e2))], (df_res_l['x_mean'] - df_res_l['s']), c='k', s=160)
plt.axhline(y=0,linestyle='--', c='k')
plt.xlabel('Trials [#]')
plt.ylabel('Diff. [A.U.]')

'''

#error bars
e2 = df_res_l_without_dup['x_SD']/np.sqrt(15) 
plt.errorbar(df_res_l_without_dup['s'], df_res_l_without_dup['x_mean'], yerr=e2, fmt='o', c='k', label='Length')

 
#linear regression
x , y = df_res_l_without_dup['s'], df_res_l_without_dup['x_mean']
m, b = np.polyfit(x, y, 1)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print(f'Length\nR square linear: {r_value**2:.3f}, \np value: {p_value:.5f} ')

plt.plot(x, m*x+b, c='k')

#unity line
lim = 20
plt.plot([0,lim], [0,lim], c='gray', linestyle='--')


plt.xlabel('Stimulus [A.U.]')
plt.ylabel('Estimate [A.U.]')

#remove top and right box and make square
sns.despine()

plt.xlim([0,lim])
plt.xticks(np.arange(0, 21, 5.0))

plt.ylim([0,lim])
plt.axis('scaled')

plt.tight_layout()
#plt.show()
plt.savefig('MOPP figures/Length.png',dpi=800)


'''
length group level plot

del df_res_l['x7']
del df_res_l['x11']

plt.clf()

import matplotlib.cm as cm
colors = cm.jet(np.linspace(0, 1, len(df_res_l['x_mean'])))
for i, c in zip(subjects,colors):
    title = 'x' + str(i)   
    plt.scatter(df_res_l['s'], df_res_l[title], edgecolor= c, facecolor= 'none', label= title[1:])

plt.scatter(df_res_l['s'], df_res_l['x_mean'], color='k')
lim = 30
plt.plot([0,lim], [0,lim],c='k')
plt.title('Line length')
plt.xlim([0,lim])
plt.ylim([0,lim])
#plt.grid()
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Stimulus [units]')
plt.ylabel('Response [units]')
plt.savefig('MOPP figures/results/Length')

'''

#numerosity, fitting polynomial regression

df_res_n['s'] = n_stimuli #same stimulus was used
df_res_n['x_mean'] = df_res_n.iloc[:, 0:-1].mean(axis=1)
df_res_n['x_SD'] = df_res_n.iloc[:, 0:-1].std(axis=1)

print(df_res_n.duplicated())
df_res_n_without_dup = df_res_n.groupby('s').mean().reset_index()

#only for fit
#visualization
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
plt.rcParams.update({'font.size': 18})
plt.clf()

df_log_n = df_res_n_without_dup[df_res_n_without_dup.s > 10]

x , y = df_log_n['s'], df_log_n['x_mean']
t = np.log(x)
p = np.polyfit(t, y, 1)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(t, y)
print(f'Numerosity\nR square log: {r_value**2:.3f} \np value: {p_value:.5f}')

a = p[0]
b = p[1]

x_fitted = np.linspace(np.min(x), np.max(x), 100)
y_fitted = a * np.log(x_fitted) + b


plt.clf()
lim = 40
plt.xlim([0,lim])
plt.ylim([0,lim])

plt.xlabel('Stimulus [dots]')
plt.ylabel('Estimate [dots]')
plt.plot([0,lim], [0,lim], c='gray', linestyle='--')
e = df_res_n_without_dup['x_SD']/np.sqrt(16)
plt.errorbar(df_res_n_without_dup['s'], df_res_n_without_dup['x_mean'], yerr=e, fmt='o', c='k', label='Numerosity')
plt.plot(x_fitted, y_fitted, c='k')

#remove top and right box and make square
sns.despine()
plt.axis('square')

#plt.legend()
#plt.show()

plt.xticks(np.arange(0, 41, 10.0))
plt.tight_layout()
plt.savefig('MOPP figures/Numerosity.png',dpi=800)


##time analysis
plt.clf()
#plt.errorbar([i for i in range(len(e2))], (df_res_l['x_mean'] - df_res_l['s'])/df_res_l['x_mean'], yerr=e2, fmt='o', c='k', label='Length')
'''
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x1'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x2'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x3'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x4'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x5'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x6'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x7'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x8'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x9'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x10'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x11'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x12'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x13'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x14'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x15'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x16'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['N-x17'] - df_res_n['s']), facecolors='none', edgecolor='gray')
plt.scatter([i for i in range(len(e2))], (df_res_n['x_mean'] - df_res_n['s']), c='k', s=160)

plt.axhline(y=0,linestyle='--', c='k')
plt.xlabel('Trials [#]')
plt.ylabel('Diff. [dots]')

'''

'''
group level plot, where x1 is participant 1 response vector
'''
md = sm.OLS(np.log(df_res_n['x_mean']), df_res_n['s']).fit()
print('Numerosity log response:')
print(md.summary())

md2 = sm.OLS(np.log(df_res_n['x_SD']), df_res_n['s']).fit()
print('Numerosity log SD:')
print(md2.summary())

#del df_res_l['x7']
#del df_res_l['x11']

md = sm.OLS(np.log(df_res_l['x_mean']), df_res_l['s']).fit()
print('Length log response:')
print(md.summary())

md2 = sm.OLS(np.log(df_res_l['x_SD']), df_res_l['s']).fit()
print('Legnth log SD:')
print(md2.summary())

#plt.scatter(df_res_n['s'], df_res_n['x_mean'], c='k')
#plt.title(f'Log[mean-x]={md.params[0]:.2f} X s, ')
#plt.show()


#plt.scatter(df_res_n['s'], df_res_n['x_SD'], c='k')
#plt.title('Log[sd-x] ` s')
#plt.show()


'''
import matplotlib.cm as cm
colors = cm.jet(np.linspace(0, 1, len(df_res_n['x_mean'])))
for i, c in zip(subjects,colors):
    title = 'x' + str(i)   
    plt.scatter(df_res_n['s'], df_res_n[title], edgecolor= c, facecolor= 'none', label= title[1:])
'''


#underestimation 

er_L = ((df_res_l['x_mean'] - df_res_l['s']) / df_res_l['s']) *100
er_n = ((df_res_n['x_mean'] - df_res_n['s']) / df_res_n['s']) *100

import scipy.stats as stats
t_statistic, p_value = stats.ttest_1samp(a=er_n, popmean=0)
print(t_statistic , p_value)
print(np.mean(er_n))



    
        
"""
The format of the columns is:
    taskID.taskName.trial.property
    e.g.,:
    0.Numerosity.0.rightAnswer = stimulus
                .answer = response
                .rt = response time
"""