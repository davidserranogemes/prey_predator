# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:28:47 2018

This file contains the object predator

@author: david
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

class PredatorAI:
    brain = 0

    #def __init__(self, board_size, num_layer,l_rate,function, n_iter):
     #   self.brain = MLPClassifier(hidden_layer_sizes=(board_size,num_layer),learning_rate='adaptative',learning_rate_init=l_rate, activation=function, max_iter=n_iter )

class Predator(object):
    _x_pos = 0
    _y_pos = 0
    
    _data_register_list = 0

    MOVE_X = 0
    MOVE_Y = 0
    
    AI = 0
    
    def __init__(self,num_epochs,num_turns):
        self._x_pos = np.ones((num_epochs,num_turns+1))
        self._y_pos = np.ones((num_epochs,num_turns+1))
        
        self.fitness = np.zeros((1,num_epochs))
        self.fitness = self.fitness[0]
        
        
        self._data_register_list = list()
        self._data_register_list.insert(0,pd.DataFrame(np.random.randint(low=0, high=10, size=(0, 38)), columns = ['Prey UP','Prey UP-RIGHT','Prey RIGHT','Prey DOWN-RIGHT','Prey DOWN','Prey DOWN-LEFT','Prey LEFT','Prey UP-LEFT','From Limit UP','From Limit RIGHT','From Limit DOWN','From Limit LEFT','Last Move Predator UP','Last Move Predator DOUBLE-UP','Last Move Predator RIGHT','Last Move Predator DOUBLE-RIGHT','Last Move Predator DOWN','Last Move Predator DOUBLE-DOWN','Last Move Predator LEFT','Last Move Predator DOUBLE-LEFT','Last Move Predator STAND','Last Move Prey UP','Last Move Prey UP-RIGHT','Last Move Prey RIGHT','Last Move Prey DOWN-RIGHT','Last Move Prey DOWN','Last Move Prey DOWN-LEFT','Last Move Prey LEFT','Last Move Prey UP-LEFT','GO UP','GO DOUBLE-UP','GO RIGHT','GO DOUBLE-RIGHT','GO DOWN','GO DOUBLE-DOWN','GO LEFT','GO DOUBLE-LEFT','STAND']))
        
        self.AI = MLPClassifier(activation = 'logistic')
        
    
    def get_X(self,epoch=0,turn=0):
        return int(self._x_pos[epoch,turn])
    
    def get_Y(self,epoch=0,turn=0):
        return int(self._y_pos[epoch,turn])
        
    def get_fitness(self,epoch):
        return self.fitness[epoch]
  
    def add_register(self, register,epoch):
        
        if len(self._data_register_list) > epoch:
            #Add the register normally
            self._data_register_list[epoch].loc[len(self._data_register_list[epoch])] = register
        else:
            if len(self._data_register_list) == epoch:
                #Create new  dataframe to the list and add the first register
                self._data_register_list.append(pd.DataFrame(np.random.randint(low=0, high=10, size=(0, 38)), columns = ['Prey UP','Prey UP-RIGHT','Prey RIGHT','Prey DOWN-RIGHT','Prey DOWN','Prey DOWN-LEFT','Prey LEFT','Prey UP-LEFT','From Limit UP','From Limit RIGHT','From Limit DOWN','From Limit LEFT','Last Move Predator UP','Last Move Predator DOUBLE-UP','Last Move Predator RIGHT','Last Move Predator DOUBLE-RIGHT','Last Move Predator DOWN','Last Move Predator DOUBLE-DOWN','Last Move Predator LEFT','Last Move Predator DOUBLE-LEFT','Last Move Predator STAND','Last Move Prey UP','Last Move Prey UP-RIGHT','Last Move Prey RIGHT','Last Move Prey DOWN-RIGHT','Last Move Prey DOWN','Last Move Prey DOWN-LEFT','Last Move Prey LEFT','Last Move Prey UP-LEFT','GO UP','GO DOUBLE-UP','GO RIGHT','GO DOUBLE-RIGHT','GO DOWN','GO DOUBLE-DOWN','GO LEFT','GO DOUBLE-LEFT','STAND']))
                self._data_register_list[epoch].loc[len(self._data_register_list[epoch])] = register
            else:
                print("Selected epoch is too big")
    
    
    def set_X(self,x,epoch=0,turn=0):
        self._x_pos[epoch,turn] = x 
    
    def set_Y(self,Y,epoch=0,turn=0):
        self._y_pos[epoch,turn] = Y

    def set_fitness(self,epoch,fitness):
        self.fitness[epoch] = fitness
        
    def select_movement(self,data_entry):
        '''
        UP = 0
        DOUBLE_UP = 0
        RIGHT = 0
        DOUBLE_RIGHT = 0
        DOWN = 0
        DOUBLE_DOWN = 0
        LEFT = 0
        DOUBLE_LEFT = 0
        STAND = 0
                    
        aux_bool = np.empty((1,9))
        aux_bool[0,0] = UP
        aux_bool[0,1] = DOUBLE_UP
        aux_bool[0,2] = RIGHT
        aux_bool[0,3] = DOUBLE_RIGHT
        aux_bool[0,4] = DOWN
        aux_bool[0,5] = DOUBLE_DOWN
        aux_bool[0,6] = LEFT
        aux_bool[0,7] = DOUBLE_LEFT
        aux_bool[0,8] = STAND
        
        return aux_bool[0]
        '''
        
        proba = self.AI.predict_proba(data_entry.reshape(1,-1))
        
        selected = np.max(proba) == proba
        selected = selected[0]
        
        predicted = np.empty((1,9))
        predicted = predicted[0]
        
        predicted[selected] = 1
        
        return predicted
        
    def prepare_register(self,data_entry,movement):
        return np.concatenate([data_entry,movement])
    
    def access_register(self,epoch):
        if epoch < len(self._data_register_list):
            return self._data_register_list[epoch]
        else:
            print("Selected epoch is too big")
    
    def move(self,epoch,turn, size_x ,size_y):
        self._x_pos[epoch,turn+1] = self._x_pos[epoch,turn] + self.MOVE_X
        self._y_pos[epoch,turn+1] = self._y_pos[epoch,turn] + self.MOVE_Y
    
        if self._x_pos[epoch,turn+1] > size_x:
            self._x_pos[epoch,turn+1] = size_x
        
        if self._y_pos[epoch,turn+1] > size_y:
            self._y_pos[epoch,turn+1] = size_y
        
        if self._x_pos[epoch,turn+1] < 0:
            self._x_pos[epoch,turn+1] = 0
        
        if self._y_pos[epoch,turn+1] < 0:
            self._y_pos[epoch,turn+1] = 0
            
            
    def train(self, first_training = False,victory = []):
        if first_training:
            aux_pd = pd.DataFrame(np.random.randint(low=0, high=1, size=(9, 38)), columns = ['Prey UP','Prey UP-RIGHT','Prey RIGHT','Prey DOWN-RIGHT','Prey DOWN','Prey DOWN-LEFT','Prey LEFT','Prey UP-LEFT','From Limit UP','From Limit RIGHT','From Limit DOWN','From Limit LEFT','Last Move Predator UP','Last Move Predator DOUBLE-UP','Last Move Predator RIGHT','Last Move Predator DOUBLE-RIGHT','Last Move Predator DOWN','Last Move Predator DOUBLE-DOWN','Last Move Predator LEFT','Last Move Predator DOUBLE-LEFT','Last Move Predator STAND','Last Move Prey UP','Last Move Prey UP-RIGHT','Last Move Prey RIGHT','Last Move Prey DOWN-RIGHT','Last Move Prey DOWN','Last Move Prey DOWN-LEFT','Last Move Prey LEFT','Last Move Prey UP-LEFT','GO UP','GO DOUBLE-UP','GO RIGHT','GO DOUBLE-RIGHT','GO DOWN','GO DOUBLE-DOWN','GO LEFT','GO DOUBLE-LEFT','STAND'])
            
            aux_pd.loc[0]['GO UP'] = 1
            aux_pd.loc[1]['GO DOUBLE-UP'] = 1
            aux_pd.loc[2]['GO RIGHT'] = 1
            aux_pd.loc[3]['GO DOUBLE-RIGHT'] = 1
            aux_pd.loc[4]['GO DOWN'] = 1
            aux_pd.loc[5]['GO DOUBLE-DOWN'] = 1
            aux_pd.loc[6]['GO LEFT'] = 1
            aux_pd.loc[7]['GO DOUBLE-LEFT'] = 1
            aux_pd.loc[8]['GO STAND'] = 1
            
            x = aux_pd[['Prey UP','Prey UP-RIGHT','Prey RIGHT','Prey DOWN-RIGHT','Prey DOWN','Prey DOWN-LEFT','Prey LEFT','Prey UP-LEFT','From Limit UP','From Limit RIGHT','From Limit DOWN','From Limit LEFT','Last Move Predator UP','Last Move Predator DOUBLE-UP','Last Move Predator RIGHT','Last Move Predator DOUBLE-RIGHT','Last Move Predator DOWN','Last Move Predator DOUBLE-DOWN','Last Move Predator LEFT','Last Move Predator DOUBLE-LEFT','Last Move Predator STAND','Last Move Prey UP','Last Move Prey UP-RIGHT','Last Move Prey RIGHT','Last Move Prey DOWN-RIGHT','Last Move Prey DOWN','Last Move Prey DOWN-LEFT','Last Move Prey LEFT','Last Move Prey UP-LEFT']]
            y = aux_pd[['GO UP','GO DOUBLE-UP','GO RIGHT','GO DOUBLE-RIGHT','GO DOWN','GO DOUBLE-DOWN','GO LEFT','GO DOUBLE-LEFT','STAND']]
            
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0)
            
            self.AI.fit(x_train,y_train)
            
        else:
            print("TODO, debe entrenar basandose en las victorias")
            aux_pd = pd.DataFrame(np.random.randint(low=0, high=1, size=(9, 38)), columns = ['Prey UP','Prey UP-RIGHT','Prey RIGHT','Prey DOWN-RIGHT','Prey DOWN','Prey DOWN-LEFT','Prey LEFT','Prey UP-LEFT','From Limit UP','From Limit RIGHT','From Limit DOWN','From Limit LEFT','Last Move Predator UP','Last Move Predator DOUBLE-UP','Last Move Predator RIGHT','Last Move Predator DOUBLE-RIGHT','Last Move Predator DOWN','Last Move Predator DOUBLE-DOWN','Last Move Predator LEFT','Last Move Predator DOUBLE-LEFT','Last Move Predator STAND','Last Move Prey UP','Last Move Prey UP-RIGHT','Last Move Prey RIGHT','Last Move Prey DOWN-RIGHT','Last Move Prey DOWN','Last Move Prey DOWN-LEFT','Last Move Prey LEFT','Last Move Prey UP-LEFT','GO UP','GO DOUBLE-UP','GO RIGHT','GO DOUBLE-RIGHT','GO DOWN','GO DOUBLE-DOWN','GO LEFT','GO DOUBLE-LEFT','STAND'])
            
            aux_pd.loc[0]['GO UP'] = 1
            aux_pd.loc[1]['GO DOUBLE-UP'] = 1
            aux_pd.loc[2]['GO RIGHT'] = 1
            aux_pd.loc[3]['GO DOUBLE-RIGHT'] = 1
            aux_pd.loc[4]['GO DOWN'] = 1
            aux_pd.loc[5]['GO DOUBLE-DOWN'] = 1
            aux_pd.loc[6]['GO LEFT'] = 1
            aux_pd.loc[7]['GO DOUBLE-LEFT'] = 1
            aux_pd.loc[8]['GO STAND'] = 1
            
            x = aux_pd[['Prey UP','Prey UP-RIGHT','Prey RIGHT','Prey DOWN-RIGHT','Prey DOWN','Prey DOWN-LEFT','Prey LEFT','Prey UP-LEFT','From Limit UP','From Limit RIGHT','From Limit DOWN','From Limit LEFT','Last Move Predator UP','Last Move Predator DOUBLE-UP','Last Move Predator RIGHT','Last Move Predator DOUBLE-RIGHT','Last Move Predator DOWN','Last Move Predator DOUBLE-DOWN','Last Move Predator LEFT','Last Move Predator DOUBLE-LEFT','Last Move Predator STAND','Last Move Prey UP','Last Move Prey UP-RIGHT','Last Move Prey RIGHT','Last Move Prey DOWN-RIGHT','Last Move Prey DOWN','Last Move Prey DOWN-LEFT','Last Move Prey LEFT','Last Move Prey UP-LEFT']]
            y = aux_pd[['GO UP','GO DOUBLE-UP','GO RIGHT','GO DOUBLE-RIGHT','GO DOWN','GO DOUBLE-DOWN','GO LEFT','GO DOUBLE-LEFT','STAND']]
            
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0)            
            
            self.AI.fit(x_train,y_train)