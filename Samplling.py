# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:44:55 2018

@author: zhaox
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


import FE_model
import FE_analysis
import uncertainty_analysis

import pyDOE


##Sampling the input parameters
lhd = pyDOE.lhs(21, samples=3000)
#sns.jointplot(lhd[:,0],lhd[:,1])

##Sampling the FEM data
parm = ((lhd-0.5)/5*4+1)*7e10  ## scale the parameters to [60%~140%]

mesh = FE_model.mesh()
properties = FE_model.properties(mesh)
BC = FE_model.boundary_condition(mesh)
FE = FE_model.FE_model(mesh, properties, BC)

analysis1 = FE_analysis.modal_analysis(FE)
analysis1.run()

FEM_parm = parm
FEM_freq = uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis=analysis1, parm=parm, target='E',index=np.arange(21))

##Sampling the test data

index=list(np.array([3,7,11,15,19])-1)
mean_test_parm=np.ones(21)*7e10
mean_test_parm[index]=np.ones(5)*6.3e10

std_test_parm=np.ones(21)*7e10*0.05
std_test_parm[index]=np.ones(5)*7e10*0.17
cov_test_parm=np.diag(std_test_parm**2)
cov_test_parm[2,6]=(7e10*0.17)**2
cov_test_parm[6,2]=(7e10*0.17)**2
cov_test_parm[14,18]=-(7e10*0.13)**2
cov_test_parm[18,14]=-(7e10*0.13)**2

parm=np.random.multivariate_normal(mean=mean_test_parm,cov=cov_test_parm,size=100)
#parm=stats.multivariate_normal(mean_test_parm,np.diag(std_test_parm**2)).rvs(size=100)
#parm=uncertainty_analysis.uncertainty_analysis.random_parm_generator(mean=mean_test_parm,std=std_test_parm,length=100)
test_parm=parm
cov_parm_test=np.cov(test_parm,rowvar=False)
test_freq=uncertainty_analysis.uncertainty_analysis.random_freq_run(analysis1,parm,target='E',index=list(np.array(np.arange(21))))

pd.DataFrame(FEM_parm).to_csv('FEM_parm.csv',index = False, header = False)
pd.DataFrame(test_parm).to_csv('test_parm.csv',index = False, header = False)
pd.DataFrame(FEM_freq).to_csv('FEM_freq.csv',index = False, header = False)
pd.DataFrame(test_freq).to_csv('test_freq.csv',index = False, header = False)
