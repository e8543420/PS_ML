# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:44:55 2018

@author: zhaox
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import FE_model
import FE_analysis
import uncertainty_analysis

import pyDOE

def flat_frf(FEM_freq,flat_way='lower_half',complex_data=True):
    if flat_way is 'lower_half':
        # Flatten the FRF data to shape[num_sample,{FRF_1_1,FRF_1_2.....}] and only the lower half
        if complex_data is True:
            FEM_frf = np.zeros((FEM_freq.shape[0],int(((FEM_freq.shape[1]+1)*FEM_freq.shape[2]/2*FEM_freq.shape[3]))),dtype=np.complex_)
            i_lower=np.triu_indices(FEM_freq.shape[1])
            for i in range(FEM_freq.shape[0]):
                FEM_frf[i] = FEM_freq[i][i_lower].flatten()
        else:
            FEM_frf = np.zeros((FEM_freq.shape[0],int(((FEM_freq.shape[1]+1)*FEM_freq.shape[2]/2*FEM_freq.shape[3]))))
            i_lower=np.triu_indices(FEM_freq.shape[1])
            for i in range(FEM_freq.shape[0]):
                FEM_frf[i] = np.abs(FEM_freq[i][i_lower]).flatten()
         # for i in range(FEM_frf.shape[0]):
         #     plt.semilogy(np.abs(FEM_frf[i,:]))
         # plt.show()
        return FEM_frf
    else:
        print('Unknown flat method!')
        exit()

##Sampling the input parameters
lhd = pyDOE.lhs(21, samples=1000)
#sns.jointplot(lhd[:,0],lhd[:,1])

##Sampling the FEM data
parm = ((lhd-0.5)/5*4+1)*7e10  ## scale the parameters to [60%~140%]

mesh = FE_model.mesh()
properties = FE_model.properties(mesh)
BC = FE_model.boundary_condition(mesh)
FE = FE_model.FE_model(mesh, properties, BC)

analysis1 = FE_analysis.modal_analysis(FE)
analysis1.FRF_run(list_points=[4,5,11,12,17,18])

FEM_parm = parm
FEM_freq = uncertainty_analysis.uncertainty_analysis.random_FRF_run(analysis=analysis1, parm=parm, target='E',index=np.arange(21))
FEM_frf = flat_frf(FEM_freq)

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
test_freq=uncertainty_analysis.uncertainty_analysis.random_FRF_run(analysis1,parm,target='E',index=list(np.array(np.arange(21))))
test_frf = flat_frf(test_freq)

pd.DataFrame(FEM_parm).to_csv('FEM_parm_frf.csv',index = False, header = False)
pd.DataFrame(test_parm).to_csv('test_parm_frf.csv',index = False, header = False)
pd.DataFrame(FEM_frf).to_csv('FEM_frf.csv',index = False, header = False)
pd.DataFrame(test_frf).to_csv('test_frf.csv',index = False, header = False)
