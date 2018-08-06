# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:19:03 2018

@author: david

This file contains the object prey
"""

from sklearn.neural_network import MLPClassifier
import numpy as np

class PreyAI:
    brain = 0

   # def __init__(self, board_size, num_layer,l_rate,function, n_iter):
    #    self.brain = MLPClassifier(hidden_layer_sizes=(board_size,num_layer),learning_rate='adaptative',learning_rate_init=l_rate, activation=function, max_iter=n_iter )

class Prey(object):
    _x_pos = 0
    _y_pos = 0
    
    _smell_range = 3
    
    def __init__(self,num_epochs,num_turns,smell_range = 3):
        self._x_pos = np.ones((num_epochs,num_turns))
        self._y_pos = np.ones((num_epochs,num_turns))
        self._smell_range = smell_range
    
    def get_X(self,epoch=0,turn=0):
        return int(self._x_pos[epoch,turn])
    
    def get_Y(self,epoch=0,turn=0):
        return int(self._y_pos[epoch,turn])
    
    def set_X(self,x,epoch=0,turn=0):
        self._x_pos[epoch,turn] = x 
    
    def set_Y(self,Y,epoch=0,turn=0):
        self._y_pos[epoch,turn] = Y
        
    def get_smell_range(self):
        return self._smell_range
    
    def set_smell_range(self,range):
        self._smell_range = range