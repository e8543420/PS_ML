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
H = analysis1.FRF_run(list_points=range(20))
# %%
H_abs = np.abs(H)
ims = []
fig, ax = plt.subplots()
ax.set_yticks(np.arange(0,H_abs.shape[0]))
ax.set_xticks(np.arange(0,H_abs.shape[0]))
ax.set_yticklabels(np.arange(1,H_abs.shape[0]+1))
ax.set_xticklabels(np.arange(1,H_abs.shape[0]+1))
for i in range(H_abs.shape[2]):
    im = ax.imshow(H_abs[:,:,i], animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
plt.show()
# ani.save('anima_FRF.mp4')
