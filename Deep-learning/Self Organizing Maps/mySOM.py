# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:17:24 2018

@author: Pranay
"""
#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data

dataset = pd.read_csv("Credit_Card_Applications.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
x = sc.fit_transform(x)

#training

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data=x, num_iteration=100)

#visualizing result

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i, X in enumerate(x):
    w = som.winner(X)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

#finding frauds
mappings = som.win_map(x)
frauds = (mappings[(6,6)])
frauds = sc.inverse_transform(frauds)
