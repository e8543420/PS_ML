# -*- coding: utf-8 -*-
"""
@author: zhaox
"""
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import FE_model
import numpy as np
import FE_analysis

#Build the FE model
mesh = FE_model.mesh()
properties = FE_model.properties(mesh)
BC = FE_model.boundary_condition(mesh)
FE = FE_model.FE_model(mesh, properties, BC)
analysis1 = FE_analysis.modal_analysis(FE)
H = analysis1.FRF_run()
for i in range(H.shape[0]):
    for j in range(i,H.shape[0]):
        plt.plot(np.abs(H[i,j,:]))
plt.show()
