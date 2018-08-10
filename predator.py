# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:28:47 2018

This file contains the object predator

@author: david
"""

from sklearn.neural_network import MLPClassifier
import numpy as np

class PredatorAI:
    brain = 0

    #def __init__(self, board_size, num_layer,l_rate,function, n_iter):
     #   self.brain = MLPClassifier(hidden_layer_sizes=(board_size,num_layer),learning_rate='adaptative',learning_rate_init=l_rate, activation=function, max_iter=n_iter )

class Predator(object):
    _x_pos = 0
    _y_pos = 0
    
    _data_register_list = 0
    
    def __init__(self,num_epochs,num_turns):
        self._x_pos = np.ones((num_epochs,num_turns))
        self._y_pos = np.ones((num_epochs,num_turns))
        
        #self._data_register_list = list()
        #self._data_register_list.insert(0,pd.DataFrame(np.random.randint(low=0, high=10, size=(0, 16)), columns = ['Predator UP','Predator RIGHT','Predator DOWN','Predator RIGHT','Limit UP','Limit RIGHT','Limit DOWN','Limit LEFT','Go UP','GO UP-RIGHT','GO RIGHT','GO DOWN-RIGHT','GO DOWN','GO DOWN-LEFT','GO LEFT','GO UP-LEFT']))
        
    
    def get_X(self,epoch=0,turn=0):
        return int(self._x_pos[epoch,turn])
    
    def get_Y(self,epoch=0,turn=0):
        return int(self._y_pos[epoch,turn])
    
    def set_X(self,x,epoch=0,turn=0):
        self._x_pos[epoch,turn] = x 
    
    def set_Y(self,Y,epoch=0,turn=0):
        self._y_pos[epoch,turn] = Y