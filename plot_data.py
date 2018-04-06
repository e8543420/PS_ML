# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:43:05 2018

@author: zhaox
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

col_name_p=['p'+str(i) for i in range(1,22)]
col_name_f=['f'+str(i) for i in range(1,21)]
FEM_parm = pd.read_csv('FEM_parm.csv',header=None,names=col_name_p)
test_parm = pd.read_csv('test_parm.csv',header=None,names=col_name_p)
FEM_freq = pd.read_csv('FEM_freq.csv',header=None,names=col_name_f)
test_freq = pd.read_csv('test_freq.csv',header=None,names=col_name_f)

#sns.jointplot(x=FEM_parm[1],y=FEM_parm[2])

## Cut the input to bins
bins=[0,0.85*7e10,1.15*7e10,3*7e10]
FEM_parm_cats=pd.DataFrame()
test_parm_cats=pd.DataFrame()
for col in FEM_parm:
    FEM_parm_cats[col]=pd.cut(FEM_parm[col],bins,labels=['lower','in','higher'])
for col in test_parm:
    test_parm_cats[col]=pd.cut(test_parm[col],bins,labels=['lower','in','higher'])

hue='p3'
data=pd.concat((test_freq[['f1','f2']],test_parm_cats[hue]),axis=1)

sns.pairplot(data, hue=hue)
plt.show()