# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:38:10 2018

@author: zhaox
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


FEM_parm = pd.read_csv('FEM_parm.csv', header=None, names=np.arange(1, 22))
test_parm = pd.read_csv(
    'test_parm.csv', header=None, names=np.arange(1, 22))
FEM_freq = pd.read_csv('FEM_freq.csv', header=None, names=np.arange(1, 21))
test_freq = pd.read_csv(
    'test_freq.csv', header=None, names=np.arange(1, 21))
# sns.jointplot(x=FEM_parm[1],y=FEM_parm[2])
# #Cut the input to bins
trus = 0.15
bins = [0, (1-trus)*7e10, (1+trus)*7e10, 3*7e10]
FEM_parm_cats = pd.DataFrame()
test_parm_cats = pd.DataFrame()
for col in FEM_parm:
    FEM_parm_cats[col] = pd.cut(FEM_parm[col], bins, labels=[
                                'lower', 'in', 'higher']).cat.codes
for col in test_parm:
    test_parm_cats[col] = pd.cut(test_parm[col], bins, labels=[
                                 'lower', 'in', 'higher']).cat.codes

# for col in test_freq:
#    test_freq[col]=pd.cut(test_freq[col],100).cat.codes
# for col in FEM_freq:
#    FEM_freq[col]=pd.cut(FEM_freq[col],100).cat.codes

mean_test_freq = test_freq.mean(axis=0).values

X = ((FEM_freq.values/mean_test_freq)-1)*10
X = np.ascontiguousarray(X, dtype=np.float32)
test_X = ((test_freq.values/mean_test_freq-1)*10)
test_X = np.ascontiguousarray(test_X, dtype=np.float32)
y = FEM_parm_cats.values
y = np.ascontiguousarray(y, dtype=np.int8)

# Scale the samples to 0 mean and 1 std
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
test_X = scaler.transform(test_X)
rng = np.random.RandomState(3)
# model=RandomForestClassifier()
model = MultiOutputClassifier(
                              SVC(
                                  kernel='rbf',
                                  cache_size=1000,
                                  random_state=rng,
                                  class_weight='balanced'
                                  ), n_jobs=1
                              )

# %%  Hyper-parameter selection
#    C_range = np.logspace(-2, 10, 13)
#    gamma_range = np.logspace(-9, 3, 13)
#    param_grid = dict(multioutputclassifier__estimator__gamma=gamma_range, multioutputclassifier__estimator__C=C_range)
#    my_pipeline = make_pipeline(model)
#    grid = GridSearchCV(my_pipeline, param_grid=param_grid,cv=3)
#    grid.fit(X,y)
#    predictions = grid.predict(test_X)
#    my_pipeline = grid

# %%  Normal pipeline
my_pipeline = make_pipeline(model)
my_pipeline.fit(X, y)
my_pipeline.set_params(multioutputclassifier__estimator__C=1,
                       multioutputclassifier__estimator__gamma='auto')
predictions = my_pipeline.predict(test_X)

# %%
results = pd.DataFrame(predictions,
                       columns=np.arange(1, 22)
                       ).apply(pd.value_counts).T
# results=test_parm_cats.apply(pd.value_counts).T
results.columns = ['lower', 'in', 'higher']
results.plot(kind='bar', stacked=True)
# pd.DataFrame(test_parm_cats,columns=np.arange(1,22)).plot(kind='hist',subplots=True)
# scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
# print(scores)
# %%

plt.figure()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (y_max-y_min)/100  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
mesh_in = np.c_[xx.ravel(), yy.ravel()]
mesh_in = np.concatenate(
    (mesh_in, np.ones((mesh_in.shape[0], 18))*X.mean(axis=0)[2:]), axis=1)

Z = my_pipeline.predict(mesh_in)[:, 2]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y[:, 2],
            cmap=plt.cm.Paired, edgecolors='k')
plt.axis('tight')
plt.show()
