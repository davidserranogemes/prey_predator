# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:19:08 2018

Base file in the prey-predator system

Initialize the board, the scores and prepare the execution


@author: david
"""

import numpy as np
import board
import prey
import predator


#CONSTANT
BOARD_SIZE = 10
NUM_PREYS = 1
NUM_PREDATORS = 1
BOARD_SIZE_X = 10
BOARD_SIZE_Y = 10


# Each epoch has X turns. If the predator catches the prey the epoch end
num_turns = 1


# Every epoch ends with a new training of the model
num_epoch = 1


# Lists of bidimensional array from np.array

# The boards have a list of at least num_turns BOARD x BOARD size.  All elements are 0 except the prey and the predator which takes -1 and +1 values
BOARD = board.Board(num_preys = NUM_PREYS,num_predators = NUM_PREDATORS,board_size_x = BOARD_SIZE_X,board_size_y = BOARD_SIZE_Y)

# Vector of winners in every epoch. TRUE the predator wins, FALSE the prey win
# We suppose that the prey wins
VICTORIES = [False] * num_epoch


###DEBUG#####

prey_test = BOARD.getPrey(id=0)
predator_test = BOARD.getPredator(id=0)

data_entry = BOARD.preparePredatorMLPData(id=0) 
movement = predator_test.select_movement(data_entry)
register = predator_test.prepare_register(data_entry,movement)
predator_test.add_register(register,epoch = 0)



####END DEBUG#####


for epoch in range(0, num_epoch):
    print(f"EPOCH: {epoch}")
    # We start the epoch
    # We create a list of boards and a list of actions for the prey and the predator.

    # In every epoch we randomize the board and the intial position of the prey and the predator.
    
    # We train the model using the previous boards as the parameters and the actions as results to predict.
    # We can use 2 different approachs. We can use all boards and actions or we can use only a percentaje of the boards.
    #The one that hasn´t succeded  doesnt train with their actual actions. 
    # They check every other actions possible and we train with those actions 
    
    # This part can be pretty expensive in computation time.

    
    # Once we have the model trained, We start the sesion. Both prey and predator move at once and we check if the prey is in a range 1-2 of the predator.
    # If this happen, the predator catches the prey and the epoch end. The predator get the maximun fitness and the prey the worst.
    
    #We start the turns. 
    for turn in range(0,num_turns):
        print(f"TURN: {turn}")
        # We save the state of the  board in the list
        
        # We ask the prey and the predator to move based on the actual board
        
        # We save the action  of both
        
        # We check if the prey has died
        
        #If it hasn´t died  we continue.
        
        #We show the move (NOT YET)
    
    # We check the  state of the board and create the fitness
    
    # We save the fitness and save who wins
    # TRUE the predator wins, FALSE the prey win
    
    # We add to the list of the action both the prey´s actions and the predator´s actions
    
    # We add to the list of the boards the list of board´s states
    
    #It ask to continue the train. If supervised
    


#We show the plot for the fitness of both the predator and the prey