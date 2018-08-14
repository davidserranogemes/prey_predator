# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:19:03 2018

@author: david

This file contains the object prey
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

class PreyAI:
    brain = 0

   # def __init__(self, board_size, num_layer,l_rate,function, n_iter):
    #    self.brain = MLPClassifier(hidden_layer_sizes=(board_size,num_layer),learning_rate='adaptative',learning_rate_init=l_rate, activation=function, max_iter=n_iter )

class Prey(object):
    _x_pos = 0
    _y_pos = 0
    
    _smell_range = 3
    
    _data_register_list = 0
    
    MOVE_X = 0
    MOVE_Y = 0
    
    
    def __init__(self,num_epochs,num_turns,smell_range = 3):
        self._x_pos = np.ones((num_epochs,num_turns+1))
        self._y_pos = np.ones((num_epochs,num_turns+1))
        self._smell_range = smell_range
        
        self._data_register_list = list()
        self._data_register_list.insert(0,pd.DataFrame(np.random.randint(low=0, high=10, size=(0, 16)), columns = ['Predator UP','Predator RIGHT','Predator DOWN','Predator LEFT','Limit UP','LimitRIGHT','Limit DOWN','Limit LEFT','Go UP','GO UP-RIGHT','GO RIGHT','GO DOWN-RIGHT','GO DOWN','GO DOWN-LEFT','GO LEFT','GO UP-LEFT']))
        
        
    def get_X(self,epoch=0,turn=0):
        return int(self._x_pos[epoch,turn])
    
    def get_Y(self,epoch=0,turn=0):
        return int(self._y_pos[epoch,turn])
    
    def get_smell_range(self):
        return self._smell_range
    
    def add_register(self, register,epoch):
       
        if len(self._data_register_list) > epoch:
            #Add the register normally
            self._data_register_list[epoch].loc[len(self._data_register_list[epoch])] = register
        else:
            if len(self._data_register_list) == epoch:
                #Create new  dataframe to the list and add the first register
                self._data_register_list.append(pd.DataFrame(np.random.randint(low=0, high=10, size=(0, 16)), columns = ['Predator UP','Predator RIGHT','Predator DOWN','Predator LEFT','Limit UP','Limit RIGHT','Limit DOWN','Limit LEFT','Go UP','GO UP-RIGHT','GO RIGHT','GO DOWN-RIGHT','GO DOWN','GO DOWN-LEFT','GO LEFT','GO UP-LEFT']))
                self._data_register_list[epoch].loc[len(self._data_register_list[epoch])] = register
            else:
                print("Selected epoch is too big")
    
    def select_movement(self,data_entry):
        UP = 0
        DOWN = 0
        LEFT = 0
        RIGHT = 0
        UP_RIGHT = 0
        UP_LEFT = 0
        DOWN_RIGHT = 0
        DOWN_LEFT = 0
                    
        aux_bool = np.empty((1,8))
        aux_bool[0,0] = UP
        aux_bool[0,1] = UP_RIGHT
        aux_bool[0,2] = RIGHT
        aux_bool[0,3] = DOWN_RIGHT
        aux_bool[0,4] = DOWN
        aux_bool[0,5] = DOWN_LEFT
        aux_bool[0,6] = LEFT
        aux_bool[0,7] = UP_LEFT
        
        return aux_bool[0]
    
    def prepare_register(self,data_entry,movement):
        return np.concatenate([data_entry,movement])
    
    def access_register(self,epoch):
        if epoch < len(self._data_register_list):
            return self._data_register_list[epoch]
        else:
            print("Selected epoch is too big")
    
    
    def set_X(self,x,epoch=0,turn=0):
        self._x_pos[epoch,turn] = x 
    
    def set_Y(self,Y,epoch=0,turn=0):
        self._y_pos[epoch,turn] = Y
        
    def set_smell_range(self,range):
        self._smell_range = range
        
    def move(self,epoch,turn):
        self._x_pos[epoch,turn+1] = self._x_pos[epoch,turn] + self.MOVE_X
        self._y_pos[epoch,turn+1] = self._y_pos[epoch,turn] + self.MOVE_Y
        