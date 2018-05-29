# -*- coding: utf-8 -*-
"""
@author: zhaox
"""

import numpy as np
import matplotlib.pyplot as plt
import FE_model

import FE_analysis

#Modal damping parameters (proportional damping)
#theta_j=alpha/(2*omiga_j)+(beta*omiga_j)/2
alpha = 1
beta = .0001

# Modal truncte order
truncate_order = 20

#the node list we want for the FRF
list_points = [5,7,10]

#Frequency list_input
freq_list = np.arange(1,1300,1)

#Build the FE model
mesh = FE_model.mesh()
properties = FE_model.properties(mesh)
BC = FE_model.boundary_condition(mesh)
FE = FE_model.FE_model(mesh, properties, BC)

analysis1 = FE_analysis.modal_analysis(FE)
[freq,modn] = analysis1.run() # modn(node_number,modal_order)
freq = np.real(freq)
# Generate the modal damping vector
modal_damping = np.zeros(freq.shape[0])
for j in range(freq.shape[0]):
    modal_damping[j] = alpha/(2*freq[j])+(beta*freq[j])/2
print(modal_damping)
# %%
H = np.zeros([len(list_points),len(list_points),len(freq_list)],dtype=np.complex_)
# Calculate the frequency response
for i_freq, value_freq in enumerate(freq_list):
    for i_input in range(len(list_points)):
        for i_output in range(len(list_points)):
            for i_order in range(truncate_order):
                H[i_input,i_output,i_freq] += (modn[i_input,i_order]*modn[i_output,i_order])/(np.square(freq[i_order])-np.square(value_freq)+2j*modal_damping[i_order]*freq[i_order]*value_freq)
plt.semilogy(H_abs[0,1])
# TODO: https://www.kaggle.com/lorinc/feature-extraction-from-images
