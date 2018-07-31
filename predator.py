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
    
    def get_X(self):
        return self._x_pos
    
    def get_Y(self):
        return self._y_pos
    
    def set_X(self,x):
        self._x_pos = x 
    
    def set_Y(self,Y):
        self._y_pos = Y