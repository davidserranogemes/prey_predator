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

NUM_PREYS = 1
NUM_PREDATORS = 1
BOARD_SIZE_X = 10
BOARD_SIZE_Y = 10


DEBUG = False
TEST = True
# Each epoch has X turns. If the predator catches the prey the epoch end
num_turns = 50


# Every epoch ends with a new training of the model
num_epoch = 100


# Lists of bidimensional array from np.array

# The boards have a list of at least num_turns BOARD x BOARD size.  All elements are 0 except the prey and the predator which takes -1 and +1 values
BOARD = board.Board(num_preys = NUM_PREYS,num_predators = NUM_PREDATORS,board_size_x = BOARD_SIZE_X,board_size_y = BOARD_SIZE_Y,num_turns=num_turns,num_epochs=num_epoch)

# Vector of winners in every epoch. TRUE the predator wins, FALSE the prey win
# We suppose that the prey wins
VICTORIES = [False] * num_epoch


###DEBUG#####



####END DEBUG#####


for epoch in range(0, num_epoch):
    print(f"Start EPOCH: {epoch}")
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
    if epoch == 0:
        BOARD.trainBoard(first_train= True)
    else:
        BOARD.trainBoard()
    #We start the turns. 
    for turn in range(0,num_turns):
        
        print(f"TURN: {turn}")
        if DEBUG:
            print(BOARD.getBoardMatrix(turn = turn,epoch = epoch))
        
        if epoch == num_epoch-1 and TEST and not DEBUG:
            print(BOARD.getBoardMatrix(turn = turn,epoch = epoch))
        
        
        # We ask the prey and the predator to move based on the actual board
        BOARD.selectMoves(epoch=epoch, turn = turn-1)
        
        #We apply the movement. This means that everyone move at the same time
        BOARD.commitMoves(epoch = epoch, turn = turn)
        
        # We check if the prey has died
        BOARD.checkPredatorKillPrey(epoch = epoch, turn = turn)        
        #If it hasn´t died  we continue.
        
        if BOARD.num_preys == 0:
            print(f"All preys has died within {turn} turns")
            break
        
        #We show the move (NOT YET)
        if DEBUG:
            input("Press Enter to continue...")
            
        if epoch == num_epoch-1 and TEST and not DEBUG:
            input("Press Enter to continue...")
    # We check the  state of the board and create the fitness
    
    # We save the fitness and save who wins
    # TRUE the predator wins, FALSE the prey win
    
    # We add to the list of the action both the prey´s actions and the predator´s actions
    
    # We add to the list of the boards the list of board´s states
    
    #It ask to continue the train. If supervised
    if epoch+1<num_epoch:
        BOARD.resetGame(epoch = epoch+1)


#We show the plot for the fitness of both the predator and the prey