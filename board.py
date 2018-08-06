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
    num_turns = 30
    num_epochs = 100
    
    turn = 0
    epoch = 0
    
    sizeX = 10
    sizeY = 10
    
    num_preys = 1
    num_predators = 1
    
    prey_vector = np.empty((1,1),dtype = prey.Prey)
    predator_vector = np.empty((1,1),dtype = predator.Predator)
    
    
    def __init__(self, num_preys = 1, num_predators = 1,board_size_x = 10,board_size_y = 10, num_turns = 30, num_epochs = 100):
        self.num_turns = num_turns
        self.num_epochs = num_epochs
        
        self.num_predators = num_predators
        self.num_preys = num_preys
        
        self.turn = 0
        self.epoch = 0
        
        self.sizeX = board_size_x
        self.sizeY = board_size_y
        
        # We create the list of preys and predators
        self.prey_vector = np.empty((1,num_preys),dtype = prey.Prey)
        self.predator_vector = np.empty((1,num_predators),dtype = predator.Predator)
    
        # I think this is quite ineficient but i dont know how to improve it
        
        self.prey_vector[:,:] = prey.Prey(num_epochs = num_epochs, num_turns=num_turns)
        self.predator_vector[:,:] = predator.Predator(num_epochs = num_epochs, num_turns=num_turns)
    
        # We  initialize all positions for every prey
        
        for i in range(0,num_preys):
            #Python has the same object for every prey right know, so it pretty useless.
            # Assign a new prey to every object in the array so they are different instances.
            
            self.prey_vector[0,i] = prey.Prey(num_epochs=num_epochs,num_turns=num_turns)
            
            x = random.randint(0,board_size_x-1)
            y = random.randint(0,board_size_y-1)
            
            self.prey_vector[0][i].set_X(x)
            self.prey_vector[0][i].set_Y(y)
        
        # We initialize all positions for every predator
        
        for i in range(0,num_predators):
            self.predator_vector[0,i] = predator.Predator(num_epochs=num_epochs,num_turns=num_turns)
            
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
        
    def getCurrentTurn(self):
        return self.turn
    
    def getCurrentEpoch(self):
        return self.epoch
    
    def getMaxTurns(self):
        return self.num_turns
    def getMaxEpoch(self):
        return self.num_epochs
        
    def getBoardDims(self):
        return self.sizeX,self.sizeY

    # This function return a matrix where the prey have the value -1 and the predators have the value +1
        
    
    def getBoardMatrix(self):
        
        matrix = np.zeros((self.sizeY+1,self.sizeX+1))
        
        
        
        
        
        
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
        
        
    def setMaxTurn(self,num_turns):
        self.num_turns = num_turns
    def setMaxEpochs(self,num_epochs):
        self.num_epochs = num_epochs
    
    def plusTurn(self):
        self.turn = self.turn +1
    def plusEpoch(self):
        self.epoch = self.epoch +1
    
    ## MLP entry data generator for preys
    def preyDetectsPredator(self, id, epoch = epoch, turn=turn):
        # Calculate the manhattan distance and if it is less or equal 3 it detects the predator.ç
        # 4 sensors 
        UP = 0
        DOWN = 0
        LEFT = 0
        RIGHT = 0
        
        aux_prey = self.getPrey(id)

        for i in range(0,self.num_predators):
            aux_predator = self.getPredator(id = i)
            
            x_difference = aux_prey.get_X(epoch = epoch,turn = turn) - aux_predator.get_X(epoch = epoch,turn = turn)
            y_difference = aux_prey.get_Y(epoch = epoch,turn = turn) - aux_predator.get_Y(epoch = epoch,turn = turn)

            manhattan_distance = np.abs(x_difference) + np.abs(y_difference)
            
            if manhattan_distance <= aux_prey.get_smell_range():
                # Thre prey can smell the predator
                if x_difference > 0 :
                    LEFT = 1
                else:
                    if x_difference < 0:
                        RIGHT = 1
                
                if y_difference > 0:
                    UP = 1
                else:
                    if y_difference < 0:
                        DOWN = 1
        
        aux_bool = np.empty((1,4))
        aux_bool[0,0] = UP
        aux_bool[0,1] = RIGHT
        aux_bool[0,2] = DOWN
        aux_bool[0,3] = LEFT
        
        return aux_bool
        
    def preyIsAtLimits(self,id,epoch = epoch, turn = turn):
        UP = 0
        DOWN = 0
        LEFT = 0
        RIGHT = 0        
        
        if self.getPrey(id).get_Y(epoch = epoch,turn = turn) == 0:
            UP = 1
        else:
            if self.getPrey(id).get_Y(epoch = epoch,turn = turn) == self.sizeY:
                DOWN = 1
        
        if self.getPrey(id).get_X(epoch = epoch,turn = turn) == 0:
            LEFT = 1
        else:
            if self.getPrey(id).get_X(epoch = epoch,turn = turn) == self.sizeX:
                RIGHT = 1
        
        aux_bool = np.empty((1,4))
        aux_bool[0,0] = UP
        aux_bool[0,1] = RIGHT
        aux_bool[0,2] = DOWN
        aux_bool[0,3] = LEFT
        
        return aux_bool
                
        
    def preparePreyMLPData(self,id,epoch = epoch,turn = turn):
        limits = self.preyIsAtLimits(id,epoch = epoch, turn = turn)
        smell = self.preyDetectsPredator(id,epoch = epoch, turn = turn)
    
        data = np.concatenate([smell[0],limits[0]])
        
        return data
    
    ## MLP entry data generator for predators
    
    def predatorHowMuchFromLimits(self,id, turn = turn, epoch = epoch):
        UP = 0
        DOWN = 0
        LEFT = 0
        RIGHT = 0
        
        UP = self.getPredator(id).get_Y(epoch = epoch,turn = turn)
        DOWN = self.sizeY - UP
        LEFT = self.getPredator(id).get_X(epoch = epoch,turn = turn)
        RIGHT = self.sizeX - LEFT
        
        aux_bool = np.empty((1,4))
        aux_bool[0,0] = UP / self.sizeY
        aux_bool[0,1] = RIGHT / self.sizeX
        aux_bool[0,2] = DOWN / self.sizeY
        aux_bool[0,3] = LEFT / self.sizeX
        
        
        
        return aux_bool
    #def predatorWhatWasLastMove(self,id,turn = turn,epoch = epoch):
        
    #def predatorWherePreyMoved(self,id,turn = turn, epoch = epoch):
        
    
        
    def predatorSeePrey(self, id,turn = turn,epoch = epoch):
        UP = 0
        DOWN = 0
        LEFT = 0
        RIGHT = 0
        UP_RIGHT = 0
        UP_LEFT = 0
        DOWN_RIGHT = 0
        DOWN_LEFT = 0
        
        
        aux_predator = self.getPredator(id = id)
        
        for i in range(0,self.num_preys):
            aux_prey = self.getPrey(id = i)
                        
            # Same X position but the prey Y pos is higher
            if aux_prey.get_X(epoch = epoch,turn = turn) == aux_predator.get_X(epoch = epoch,turn = turn) and aux_prey.get_Y(epoch = epoch,turn = turn) > aux_predator.get_Y(epoch = epoch,turn = turn):
                DOWN = 1
            
            
            else:
                # Same X position but the prey Y pos is lower    
                if aux_prey.get_X(epoch = epoch,turn = turn) == aux_predator.get_X(epoch = epoch,turn = turn) and aux_prey.get_Y(epoch = epoch,turn = turn) < aux_predator.get_Y(epoch = epoch,turn = turn):
                    UP = 1
                else:
                    
                    # Same Y position but the prey X pos is higher
                    if aux_prey.get_Y(epoch = epoch,turn = turn) == aux_predator.get_Y(epoch = epoch,turn = turn) and aux_prey.get_X(epoch = epoch,turn = turn) > aux_predator.get_X(epoch = epoch,turn = turn):
                        RIGHT = 1
                    else:
                        
                        # Same Y position but the prey X pos is lower
                        if aux_prey.get_Y(epoch = epoch,turn = turn) == aux_predator.get_Y(epoch = epoch,turn = turn) and aux_prey.get_X(epoch = epoch,turn = turn) < aux_predator.get_X(epoch = epoch,turn = turn):
                            LEFT = 1
                        else:
                            # The 
                            
                            X_diff = aux_prey.get_X(epoch = epoch,turn = turn) - aux_predator.get_X(epoch = epoch,turn = turn)
                            Y_diff = aux_prey.get_Y(epoch = epoch,turn = turn) - aux_predator.get_Y(epoch = epoch,turn = turn)
                            
                            if X_diff > 0 and Y_diff > 0 and np.abs(X_diff)==np.abs(Y_diff):
                                DOWN_RIGHT = 1
                            else:
                                if X_diff > 0 and Y_diff < 0 and np.abs(X_diff)==np.abs(Y_diff):
                                    UP_RIGHT = 1
                                else:
                                    if X_diff < 0 and Y_diff < 0 and np.abs(X_diff)==np.abs(Y_diff):
                                        UP_LEFT = 1
                                    else:
                                        if X_diff < 0 and Y_diff > 0 and np.abs(X_diff)==np.abs(Y_diff):
                                            DOWN_LEFT = 1
                                        
                        
                            
                        
                
            
        
             
        aux_bool = np.empty((1,8))
        aux_bool[0,0] = UP
        aux_bool[0,1] = UP_RIGHT
        aux_bool[0,2] = RIGHT
        aux_bool[0,3] = DOWN_RIGHT
        aux_bool[0,4] = DOWN
        aux_bool[0,5] = DOWN_LEFT
        aux_bool[0,6] = LEFT
        aux_bool[0,7] = UP_LEFT
        
        return aux_bool