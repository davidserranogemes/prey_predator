# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:19:08 2018

Base file in the prey-predator system

Initialize the board, the scores and prepare the execution


@author: david
"""

import numpy as np

BOARD_SIZE = 10


# Each epoch has X turns. If the predator catches the prey the epoch end
num_turns = 100


# Every epoch ends with a new training of the model
num_epoch = 200


# Lists of bidimensional array from np.array

# The boards have a list of at least num_turns BOARD x BOARD size.  All elements are 0 except the prey and the predator which takes -1 and +1 values

ALL_PREY_ACTIONS = list()
ALL_PREDATOR_ACTIONS = list()

# Array of prey´s fitness and the predator´s fitness
ALL_PREDATOR_FITNESS = np.zeros(1,num_epoch)
ALL_PREY_FITNESS = np.zeros(1,num_epoch)

# Vector of winners in every epoch. TRUE the predator wins, FALSE the prey win
# We suppose that the prey wins
VICTORIES = [False] * num_epoch

for epoch in range(0, num_epoch):
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