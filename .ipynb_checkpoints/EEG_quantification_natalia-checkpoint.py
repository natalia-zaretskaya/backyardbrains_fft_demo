#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:17:19 2023

@author: natalia
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import csv
import pandas as pd
#file = '/Users/natalia/Music/Spike Recorder/BYB_Recording_2023-03-14_09.33.00.wav'
file = 'Z:/owncloud0/work/teaching/2023_SS_Hands-on_EEG/coding/BYB_Recording_2023-03-14_09.33.00.wav'
fs, data = waves.read(file)

length_data=np.shape(data)
length_new=length_data[0]*0.05
ld_int=int(length_new)

from scipy import signal
data_new=signal.resample(data,ld_int)

# the code below has to be run at once
plt.figure('Spectrogram')
d, f, t, im = plt.specgram(data_new, NFFT= 256, Fs=500, noverlap=250)
plt.ylim(0, 90)
plt.colorbar(label= "Power/Frequency")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.show()

# save frequences as CSV
matrixf=np.array(f).T
np.savetxt('Frequencies.csv', matrixf)
df = pd.read_csv("Frequencies.csv", header=None, index_col=None)
df.columns = ["Frequencies"]
df.to_csv("Frequencies.csv", index=False)

# select the alpha frequency
position_vector=[]
length_f=np.shape(f)
l_row_f=length_f[0]
for i in range(0, l_row_f):
    if f[i]>=7 and f[i]<=12:
        position_vector.append(i)

# extract alpha over time, averaging over frequencies
length_d=np.shape(d)
l_col_d=length_d[1]
AlphaRange=[]
for i in range(0,l_col_d):
    AlphaRange.append(np.mean(d[position_vector[0]:max(position_vector)+1,i]))
    
# smooth the data    
def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

# plot alpha
plt.figure('AlphaRange')
y=smoothTriangle(AlphaRange, 100)
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.xlim(0,max(t))

# save alpha as CSV
datosy=np.asarray(y)
datosyt=np.array(
       [
       datosy,
        t
        ])
with open ('datosyt.csv', 'w', newline='') as file:
   writer=csv.writer(file, dialect='excel-tab')
   writer.writerows(datosyt.T)

df = pd.read_csv("datosyt.csv", header=None, index_col=None)
df.columns = ["Power                   Time"]
df.to_csv("datosyt.csv", index=False)

# now we need the vector with events
#event_file = '/Users/natalia/Music/Spike Recorder/BYB_Recording_2023-03-14_09.33.00-events.txt'
event_file = 'Z:/owncloud0/work/teaching/2023_SS_Hands-on_EEG/coding/BYB_Recording_2023-03-14_09.33.00-events.txt'
event_data = np.loadtxt(event_file, comments='#', delimiter=',')
event_times = event_data[:,1]
tg = event_times.tolist()
tg.append(max(t)+1)

length_t=np.shape(t)
l_row_t=length_t[0]
eyesclosed=[]
eyesopen=[]
j=0  #initial variable to traverse tg
l=0  #initial variable to loop through the "y" data
for i in range(0, l_row_t):
    if t[i]>=tg[j]:
        
        if j%2==0:
            eyesopen.append(np.mean(datosy[l:i]))
        if j%2==1:
            eyesclosed.append(np.mean(datosy[l:i]))
        l=i
        j=j+1

        
plt.figure('DataAnalysis')
plt.boxplot([eyesopen, eyesclosed], sym = 'ko', whis = 1.5)
plt.xticks([1,2], ['Eyes open', 'Eyes closed'], size = 'small', color = 'k')
plt.ylabel('AlphaPower')

meanopen=np.mean(eyesopen)
meanclosed=np.mean(eyesclosed)
sdopen=np.std(eyesopen)
sdclosed=np.std(eyesclosed)
eyes=np.array([eyesopen, eyesclosed])

from scipy import stats
result=stats.ttest_ind(eyesopen, eyesclosed, equal_var = False)
print(result)

