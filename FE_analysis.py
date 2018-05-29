# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:15:37 2017

@author: zhaox
"""
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import FE_model

class modal_analysis:
    freq=None
    modn=None
    def __init__(self,FE_model):
        self.FE_model=FE_model

    def run(self):
        ActiveDof=self.FE_model.ActiveDof
        K=self.FE_model.K
        M=self.FE_model.M
        K=K[list(ActiveDof)][:,list(ActiveDof)]
        M=M[list(ActiveDof)][:,list(ActiveDof)]

        [valn,modn]=eig(K,M)
        freq=np.sqrt(valn)/(2*np.pi)
        freq=freq[np.argsort(freq)]
        modn=modn[np.argsort(freq)]
        self.freq=freq
        self.modn=modn
        return freq,modn
    def plot(self,mode=1,sf=2):
        """
        #--------------Plot modal shapes-------------------
        #analysis2.plot(mode=10,sf=1.1)
        """
        mode-=1 #deal with numbering
        N=len(self.FE_model.ActiveDof)
        uu=range(0,N,2)
        vv=range(1,N,2)
        plt.figure()

        nodeCoord=self.FE_model.nodeCoord
        elementNodes=self.FE_model.elementNodes
        modn=self.modn
        freq=self.freq
        xx=np.append([0,0],nodeCoord[2:][:,0]+sf*modn[uu][:,mode])
        yy=np.append([0,1],nodeCoord[2:][:,1]+sf*modn[vv][:,mode])
        plt.scatter(xx,yy,color='blue')
        plt.scatter(nodeCoord[:,0],nodeCoord[:,1],color='red')

        for point_id in elementNodes:
            line_x=np.append(nodeCoord[point_id[0]][0],nodeCoord[point_id[1]][0])
            line_y=np.append(nodeCoord[point_id[0]][1],nodeCoord[point_id[1]][1])
            plt.plot(line_x,line_y,color='red')

        for point_id in elementNodes:
            line_x=np.append(xx[point_id[0]],xx[point_id[1]])
            line_y=np.append(yy[point_id[0]],yy[point_id[1]])
            plt.plot(line_x,line_y,color='blue')

        plt.title('Plot for the '+str(mode+1)+'th modal shape (freq='+str(round(freq[mode].real,2))+'Hz)')
    def sensi_freq(self,step=0.01,target='E',index=None):
        """
        #-----------------Display sensi matrix------------------
        #freq,modn=analysis1.reanalysis(target='E',index=list(np.array([2,8])-1),data=np.ones(2)*6.3e10)
        #sensi=analysis1.sensi_freq(step=0.01,target='E',index=list(np.array([2,8])-1))
        #analysis1.run()
        #analysis1.plot(mode=10,sf=1.1)
        #
        """
        if self.freq is None:
            self.run()
        num_freq=len(self.freq)
        num_parm=len(index)
        sensi=np.matrix(np.zeros([num_freq,num_parm]))
        for i,i_parm in enumerate(index):
            parm1=self.FE_model.properties.E[i_parm]
            parm2=self.FE_model.properties.E[i_parm]*(1+step)
            properties2=self.FE_model.properties.modify(target=target,index=i_parm,data=parm2)
            FE2=FE_model.FE_model(self.FE_model.mesh,properties2,self.FE_model.boundary_condition)
            analysis2=modal_analysis(FE2)
            analysis2.run()
            freq2=analysis2.freq
            D_freq=(freq2-self.freq).real
            sensi[:,i]=np.matrix(D_freq/(parm2-parm1)).T
        return sensi

    def reanalysis(self,target='E',index=None,data=None):
        if self.freq is None:
            self.run()
        properties2=self.FE_model.properties.modify(target=target,index=index,data=data)
        FE2=FE_model.FE_model(self.FE_model.mesh,properties2,self.FE_model.boundary_condition)
        analysis2=modal_analysis(FE2)
        freq,modn=analysis2.run()
        return freq.real,modn
    def FRF_run(self,truncate_order=20,list_points=[4,9,14,19],freq_list = np.arange(1,1300,1),alpha=1,beta=.0001):
        if self.freq is None:
            self.run()
        modn = self.modn
        freq = self.freq
        freq = np.real(freq)
        # Generate the modal damping vector
        modal_damping = np.zeros(freq.shape[0])
        for j in range(freq.shape[0]):
            modal_damping[j] = alpha/(2*freq[j])+(beta*freq[j])/2
        # %%
        H = np.zeros([len(list_points),len(list_points),len(freq_list)],dtype=np.complex_)
        # Calculate the frequency response
        for i_freq, value_freq in enumerate(freq_list):
            for i_input in range(len(list_points)):
                for i_output in range(i_input,len(list_points)):
                    for i_order in range(truncate_order):
                        H[i_input,i_output,i_freq] += (modn[i_input,i_order]*modn[i_output,i_order])/(np.square(freq[i_order])-np.square(value_freq)+2j*modal_damping[i_order]*freq[i_order]*value_freq)
        i_lower = np.tril_indices(H.shape[0], -1)
        for i in range(H.shape[2]):
            H[:,:,i][i_lower] = H[:,:,i].T[i_lower]
        return H
