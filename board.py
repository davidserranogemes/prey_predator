# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 23:26:01 2018

@author: david
"""

import numpy as np
import prey
import predator
import random

class Board(object):
    #They have the X position in the X 
    prey_vector = np.empty((1,1),dtype = prey.Prey)
    predator_vector = np.empty((1,1),dtype = predator.Predator)
    sizeX = 10
    sizeY = 10
    
    def __init__(self, num_preys = 1, num_predators = 1,board_size_x = 10,board_size_y = 10):
        self.sizeX = board_size_x
        self.sizeY = board_size_y
        
        # We create the list of preys and predators
        self.prey_vector = np.empty((1,num_preys),dtype = prey.Prey)
        self.predator_vector = np.empty((1,num_predators),dtype = predator.Predator)
    
        # I think this is quite ineficient but i dont know how to improve it
        
        self.prey_vector[:,:] = prey.Prey()
        self.predator_vector[:,:] = predator.Predator()
    
        # We  initialize all positions for every prey
        
        for i in range(0,num_preys):
            #Python has the same object for every prey right know, so it pretty useless.
            # Assign a new prey to every object in the array so they are different instances.
            
            self.prey_vector[0,i] = prey.Prey()
            
            x = random.randint(0,board_size_x-1)
            y = random.randint(0,board_size_y-1)
            
            self.prey_vector[0][i].set_X(x)
            self.prey_vector[0][i].set_Y(y)
        
        # We initialize all positions for every predator
        
        for i in range(0,num_predators):
            self.predator_vector[0,i] = predator.Predator()
            
            x = random.randint(0,board_size_x-1)
            y = random.randint(0,board_size_y-1)
            
            self.predator_vector[0,i].set_X(x)
            self.predator_vector[0,i].set_Y(y)



    ### -------------------- Observers---------###
    def getPrey(self,id):
        if(id<self.prey_vector.size and id>=0):
            return self.prey_vector[0,id]
        else:
            return None
    
    def getPredator(self, id):
        if(id<self.predator_vector.size and id>=0):
            return self.predator_vector[0,id]
        else:
            return None
    def getBoardDims(self):
        return self.sizeX,self.sizeY

    # This function return a matrix where the prey have the value -1 and the predators have the value +1
    def getBoardMatrix(self):
        
        matrix = np.zeros((self.sizeY,self.sizeX))
         
        for i in range(0,self.prey_vector.size):
            matrix[self.prey_vector[0,i].get_Y(),self.prey_vector[0,i].get_X()] = -1
         
        for i in range(0,self.predator_vector.size):
            matrix[self.predator_vector[0,i].get_Y(),self.predator_vector[0,i].get_X()] = 1
        
        return matrix
    
    
    
    ###------ Action functions------###
    #This function try to separate all preys from every predator, so the predator cant kill them instantly.
    #TODO
    def correctBoardInitPositions():
        print("TOD0")