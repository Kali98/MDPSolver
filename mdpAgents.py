# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py
from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import math

# Using Grid class that maps out pacman's environment from 6CCS3AIN. Week 5 - Temporal Probabilistic Reasoning. https://keats.kcl.ac.uk/course/view.php?id=66991. Last accessed 2nd Nov 2019.
#
#
# The map itself is implemented as a nested list, and the interface
# allows it to be accessed by specifying x, y locations.
#
class Grid:       
    # Constructor
    #
    # 
    #
    # grid:   an array that has one position for each element in the grid.
    # width:  the width of the grid
    # height: the height of the grid
    #
    def __init__(self, width, height):
        self.width = width
        self.height = height
        subgrid = []
        for i in range(self.height):
            row=[]
            for j in range(self.width):
                row.append(0)
            subgrid.append(row)
        self.grid = subgrid

    # Print the grid out.
    def display(self):       
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print self.grid[i][j],
            # A new line after each line of the grid
            print 
        # A line after the grid
        print

    # The display function prints the grid out upside down. This
    # prints the grid out so that it matches the view we see when we
    # look at Pacman.
    def prettyDisplay(self):       
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print self.grid[self.height - (i + 1)][j],
            # A new line after each line of the grid
            print 
        # A line after the grid
        print
        
    # Set and get the values of specific elements in the grid.
    # Here x and y are indices.
    def setValue(self, x, y, value):
        self.grid[y][x] = value

    def getValue(self, x, y):
        return self.grid[y][x]

    # Return width and height to support functions that manipulate the
    # values stored in the grid.
    def getHeight(self):
        return self.height

    def getWidth(self):
        return self.width
#
# An agent that creates a map.
#
# As currently implemented, the map places a % for each section of
# wall, a * where there is food, and a space character otherwise. I don't change 
# values, instead I create my own array of costs, rewards, utility values etc. for the game based
# on the information on the map (personally just more convenient for me)

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        self.foodCost = 1.5 # Food cost variable corresponds to the cost of all cells that have food items in them
        self.emptyTileCost = -0.2 # Empty tile variable corresponds to the cost of all cells that are empty
        self.ghostCost = -10 # Ghost cost variable corresponds to the cost of cells that have a ghost in them
        self.edibleGhostCost = 0 # Edible ghost cost variable corresponds to the cells that have edible ghosts in them

        self.expectedUtilities = [] # Stores the collection of all expected Utilities for each cell (post-value-iteration process)
        self.coordinatesOfEachEXField = [] # Stores coordinates that correspond to the locations of the expected utilities (same indexes as self.expectedUtilities)
        self.iterationCount = 0 # keeps count of the number of iterations until the values in self.expectedUtilities stop changing (used only for display)
        self.gamma = 0.925 # Gamma variable is the discount factor (between value 0 and 1) and models the preference of the agent for current over future rewards

    # Gets run after an MDPAgent object is created and assigns initial state and costs (initial state of the value iteration process)
    def registerInitialState(self, state):
         # Makes a map of the right size (according to the grid)
         self.makeMap(state)
         self.addWallsToMap(state)
         self.updateFoodInMap(state)
         self.map.display()

         MDPAgent.setInitialStateInfo(self,state, self.foodCost, 0) # sets initial state for the expected utilities (all cells starting from zero except food cells)

    # Final function to keep the costs the same between games and reset the expected utility values to their initial state
    def final(self, state):
        self.foodCost = 1.5
        self.emptyTileCost = -0.2
        self.ghostCost = -10
        self.edibleGhostCost = 0

        self.expectedUtilities = []
        self.coordinatesOfEachEXField = []
        self.iterationCount = 0
        self.gamma = 0.925
        MDPAgent.setInitialStateInfo(self,state, self.foodCost, 0)

    # Make a map by creating a grid of the right size
    def makeMap(self,state):
        corners = api.corners(state)
        print corners
        height = self.getLayoutHeight(corners)
        width  = self.getLayoutWidth(corners)
        self.map = Grid(width, height)
        
    # Functions to get the height and the width of the grid.
    #
    # We add one to the value returned by corners to switch from the
    # index (returned by corners) to the size of the grid
    def getLayoutHeight(self, corners):
        height = -1
        for i in range(len(corners)):
            if corners[i][1] > height:
                height = corners[i][1]
        return height + 1

    def getLayoutWidth(self, corners):
        width = -1
        for i in range(len(corners)):
            if corners[i][0] > width:
                width = corners[i][0]
        return width + 1

    # Functions to manipulate the map.
    #
    # Put every element in the list of wall elements into the map
    def addWallsToMap(self, state):
        walls = api.walls(state)
        for i in range(len(walls)):
            self.map.setValue(walls[i][0], walls[i][1], '%')

    # Create a map with a current picture of the food that exists.
    def updateFoodInMap(self, state):
        # First, make all grid elements that aren't walls blank.
        for i in range(self.map.getWidth()):
            for j in range(self.map.getHeight()):
                if self.map.getValue(i, j) != '%':
                    self.map.setValue(i, j, ' ')
        food = api.food(state)
        for i in range(len(food)):
            self.map.setValue(food[i][0], food[i][1], '*')

    # Functions for MDP Agent
    #
    #Function setInitialStateInfo sets initial state for value iteration
    def setInitialStateInfo(self,state,foodCost,otherCost): 
        #Nested For loop to iterate through the whole grid
        for i in range(self.map.getHeight()):
            for j in range(self.map.getWidth()): 
                # Using the map values; I check if a cell is empty or if it has a food item in it and then I populate the expected utilities collection with
                # appropriate values for the initial state
                if self.map.getValue(j,i) == " ": 
                    self.expectedUtilities.append(otherCost)
                    self.coordinatesOfEachEXField.append((j,i))
                elif self.map.getValue(j,i) == "*" :
                    self.expectedUtilities.append(foodCost)
                    self.coordinatesOfEachEXField.append((j,i))

    #Function calculateNextValueIteration retrieves next iteration values of expected utility
    def calculateNextIterationValues(self,state):
        nextExpectedUtilities = [] # This array will store the return value of all next iteration expected utility values

        for i in range(len(self.expectedUtilities)): # For loop to iterate through all current expected utility values
            # For the current expected utility value and it's location; extract the map value and the location of a cell to the west
            westValue = self.map.getValue(self.coordinatesOfEachEXField[i][0]-1,self.coordinatesOfEachEXField[i][1]) 
            westValueCoordinates = (self.coordinatesOfEachEXField[i][0]-1,self.coordinatesOfEachEXField[i][1])
            # For the current expected utility value and it's location; extract the map value and the location of a cell to the east
            eastValue = self.map.getValue(self.coordinatesOfEachEXField[i][0]+1,self.coordinatesOfEachEXField[i][1])
            eastValueCoordinates = (self.coordinatesOfEachEXField[i][0]+1,self.coordinatesOfEachEXField[i][1])
            # For the current expected utility value and it's location; extract the map value and the location of a cell to the south
            southValue = self.map.getValue(self.coordinatesOfEachEXField[i][0],self.coordinatesOfEachEXField[i][1]-1)
            southValueCoordinates= (self.coordinatesOfEachEXField[i][0],self.coordinatesOfEachEXField[i][1]-1)
            # For the current expected utility value and it's location; extract the map value and the location of a cell to the north
            northValue = self.map.getValue(self.coordinatesOfEachEXField[i][0],self.coordinatesOfEachEXField[i][1]+1)
            northValueCoordinates = (self.coordinatesOfEachEXField[i][0],self.coordinatesOfEachEXField[i][1]+1)

            # leftCalc1, leftCalc2 and leftCalc3 calculate the utility of going left from this current location for example if the state was (3,1) 
            # then leftCalc1 variable will be storing the value of: 0.8U(2,1), leftCalc2 would be storing the value of: 0.1U(3,1) assuming theres 
            # a wall at (3,0) and the same procedure is carried out for (Right), (Down) and (Up) to gather utility values for all directions

            leftCalc1 = 0.8 * MDPAgent.getCostFactor(self,state,westValue,"West", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)
            leftCalc2 = 0.1 * MDPAgent.getCostFactor(self,state,southValue,"South", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)
            leftCalc3 = 0.1 * MDPAgent.getCostFactor(self,state,northValue,"North", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)

            rightCalc1 = 0.8 * MDPAgent.getCostFactor(self,state,eastValue,"East", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)
            rightCalc2 = 0.1 * MDPAgent.getCostFactor(self,state,northValue,"North", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)
            rightCalc3 = 0.1 * MDPAgent.getCostFactor(self,state,southValue,"South", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)

            downCalc1 = 0.8 * MDPAgent.getCostFactor(self,state,southValue,"South", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)
            downCalc2 = 0.1 * MDPAgent.getCostFactor(self,state,eastValue,"East", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)
            downCalc3 = 0.1 * MDPAgent.getCostFactor(self,state,westValue,"West", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)

            upCalc1 = 0.8 * MDPAgent.getCostFactor(self,state,northValue,"North", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)
            upCalc2 = 0.1 * MDPAgent.getCostFactor(self,state,westValue,"West", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)
            upCalc3 = 0.1 * MDPAgent.getCostFactor(self,state,eastValue,"East", self.expectedUtilities[i], westValueCoordinates, eastValueCoordinates, southValueCoordinates, northValueCoordinates)

            # All gathered values are added and stored inside potentialUtilityValues array
            potentialUtilityValues = [(upCalc1+upCalc2+upCalc3), (leftCalc1+leftCalc2+leftCalc3), (downCalc1+downCalc2+downCalc3), (rightCalc1+rightCalc2+rightCalc3)] 
            # Max utility value is extracted and stored in a variable
            maxUtilityValue = max(potentialUtilityValues)           
            
            utilityOfTheState = self.emptyTileCost + (self.gamma * maxUtilityValue) # Bellman update is applied [Ui+1(s) <--- R(s) + y max...] to the current state
            #utilityOfTheState = round(utilityOfTheState,3) # Value is rounded to three decimal places
           
            ghostInfo = api.ghostStates(state) # Retrieving ghost information
            numOfGhosts = len(ghostInfo) # Number of ghosts

            if self.map.getValue(self.coordinatesOfEachEXField[i][0], self.coordinatesOfEachEXField[i][1]) == "*" or self.map.getValue(self.coordinatesOfEachEXField[i][0], self.coordinatesOfEachEXField[i][1]) == " ":
                # IF logic to establish if theres a ghost within the same cell of a food or empty cell (comparing ghost coordinates to the current state coordinates)
                if (ghostInfo[0][0][0],ghostInfo[0][0][1]) == (self.coordinatesOfEachEXField[i][0], self.coordinatesOfEachEXField[i][1]) or (ghostInfo[numOfGhosts-1][0][0],ghostInfo[numOfGhosts-1][0][1]) == (self.coordinatesOfEachEXField[i][0], self.coordinatesOfEachEXField[i][1]) :
                    # if theres a ghost at the current state; checking if it's edible or not
                    if ghostInfo[0][1] == 1 and ghostInfo[numOfGhosts-1][1] == 1:
                        # if it's edible then assign self.edibleGhostCost cost to the state's utility regardless if the same cell has a food or if it's empty
                        nextExpectedUtilities.append(self.edibleGhostCost)
                    else :
                        # else assign the normal ghost cost utility to the state
                        nextExpectedUtilities.append(self.ghostCost)
                else :  
                    if numOfGhosts == 2 :
                        # if the ghosts are not located within the current state; I check all food and empty cells that are within a certain
                        # distance threshold to the ghost. Below I use a distance formula [(x2 - x1)^2 + (y2 - y1)^2] to determine the distance 
                        # of the current state to the ghost
                        distanceToGhost1 = abs(self.coordinatesOfEachEXField[i][0] - ghostInfo[0][0][0])**2 + abs(self.coordinatesOfEachEXField[i][1]- ghostInfo[0][0][1])**2
                        distanceToGhost2 = abs(self.coordinatesOfEachEXField[i][0] - ghostInfo[1][0][0])**2 + abs(self.coordinatesOfEachEXField[i][1]- ghostInfo[1][0][1])**2
                    else :
                        # In the case of there being only one ghost; I set the second ghost distance variable to something will be ignored
                        distanceToGhost1 = abs(self.coordinatesOfEachEXField[i][0] - ghostInfo[0][0][0])**2 + abs(self.coordinatesOfEachEXField[i][1]- ghostInfo[0][0][1])**2
                        distanceToGhost2 = 10000
                    
                    

                    if self.map.getValue(self.coordinatesOfEachEXField[i][0], self.coordinatesOfEachEXField[i][1]) == "*" :  # If the value of the current state is a food
                        
                        # Note I don't square root the distance values because when square rooting there's too much a difference between 1 and 2 blocks for the smallGrid
                        avoidDistance = 0
                        # depending on the grid scale; I set the avoid distance smaller if it's below or equal to 7x7 because pacman doesn't have as much space to move around
                        # in smaller grids to avoid the ghost/ghosts
                        if self.map.getHeight() <= 7 or self.map.getWidth() <= 7:
                            avoidDistance = 2.5
                        else :
                            avoidDistance = 5.5
                        
                        
                        if distanceToGhost1 <= avoidDistance or distanceToGhost2 <= avoidDistance : 
                            nextExpectedUtilities.append(utilityOfTheState)  #  assign the calculated utility based on the ghost being next to it  
                        else : 
                            nextExpectedUtilities.append(self.foodCost)  # else asign the normal food cost 
                    else :  
                        nextExpectedUtilities.append(utilityOfTheState)  # otherwise if it is an empty cell then assign  the calculated utility
        return nextExpectedUtilities    # return the next iteration of values (when the for loop terminates, all expected utilities iterate)
    
    #Function getCostFactor get's a multiplication value based on what is next to pacman [used to workout utilities of (left), (right), (down) and (up)]
    def getCostFactor(self,state,stateValue, directionOfMove, currentExpectedUtility,  westValCord, eastValCord, southValCord, northValCord):

        multiplicationFactor = 0 # the value that's returned is first set to 0

        if stateValue == "%" : # if the value of the next state is a wall
            # the multiplication factor is the utility of the same cell where pacman is located because pacman will not move locations if he hits a wall
            multiplicationFactor = currentExpectedUtility 
            # if the value of next state empty
        elif stateValue == " " : 
            # assign the expected utlity of the state that's empty based on the direction that pacman would be moving    
            multiplicationFactor = MDPAgent.getNeighbourCost(self,state, directionOfMove, westValCord, eastValCord, southValCord, northValCord)
            #if the value of next state is a food
        elif stateValue == "*" :
            # assign the expected utlity of the state that's a food based on the direction that pacman would be moving
            multiplicationFactor = MDPAgent.getNeighbourCost(self,state, directionOfMove, westValCord, eastValCord, southValCord, northValCord)
        return multiplicationFactor

    #Function getNeighbourCost supports the getCostFactor function by returning the expected utility value of a cell based on coordinates 
    def getNeighbourCost(self,state, neighbourDirection, westValCord, eastValCord, southValCord, northValCord) :
        neighbourCost = 0 # return value that I first set to 0
        indexUtilityOfState = 0 # index used to find the expected utility of a state based on location

        if neighbourDirection == "West" : # if the direction of the state is west
            indexUtilityOfState = self.coordinatesOfEachEXField.index(westValCord) # find index of the west utility state cost
            neighbourCost = self.expectedUtilities[indexUtilityOfState] # utility value is assigned to the neigbhour expected utility value
        #same is carried out for the East, South and North locations
        elif neighbourDirection == "East" :  
            indexUtilityOfState = self.coordinatesOfEachEXField.index(eastValCord)
            neighbourCost = self.expectedUtilities[indexUtilityOfState]

        elif neighbourDirection == "South" : 
            indexUtilityOfState = self.coordinatesOfEachEXField.index(southValCord)       
            neighbourCost = self.expectedUtilities[indexUtilityOfState] 

        elif neighbourDirection == "North" :  
            indexUtilityOfState = self.coordinatesOfEachEXField.index(northValCord)         
            neighbourCost = self.expectedUtilities[indexUtilityOfState] 
            
        return neighbourCost
        
    def getAction(self, state):
        #Update Map and display every state
        self.updateFoodInMap(state)
        # self.map.prettyDisplay() displaying the map in console      
        # Remove STOP as a legal action as it will not be a necessary action at any state during the game
        legal = api.legalActions(state)
        pacman = api.whereAmI(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        print
        #print ("Expected Utilities for all states:", self.expectedUtilities)
        #print ("Coordinates of expected Utilities for all states:", self.coordinatesOfEachEXField)
       
        self.iterationCount = 0
        utilityChangeCheck = [] # Used to store the expected utilities of a previous iteration
        
        while (self.iterationCount < 30) : # while loop to keep iterating the values until the values in self.expectedUtilities no longer change or if the number of iterations reaches 50
            utilityChangeCheck = self.expectedUtilities
            self.expectedUtilities = MDPAgent.calculateNextIterationValues(self,state)  # calling calculateNextIterationValues function to get next iteration of values
            self.iterationCount += 1 # iteration the iteration count
            if(self.expectedUtilities == utilityChangeCheck): # added a threshold to stop iterating at a certain threshold to speed up the performance
                break

        print ("number of iterations carried out:", self.iterationCount)  
        
        possibleExpectedUtilityMoves = [] # array that will store possible expected utlities based on the legal moves that pacman can carry out
        moveIndexes = [] # array that will store indexes for the moves for each expected utility 
        for i in range(len(legal)):  # iterating through all legal moves
            if legal[i] == "West":
                westCell = (pacman[0]-1, pacman[1])  #westCell variable to establish a tuple (x,y) which is to the west of pacman
                expectedUtilityOfMoveIndex = self.coordinatesOfEachEXField.index(westCell) # get index based on the coordinates of a expected utility of the west cell
                possibleExpectedUtilityMoves.append(self.expectedUtilities[expectedUtilityOfMoveIndex]) # expected utility of the west cell of pacman stored in possible expected utility moves
                moveIndexes.append("West") # under the same index, a move reference is stored which in this case is "West"
            # same procedure carried out in this If logic to establish the expected utilities and possible moves that pacman can carry out
            elif legal[i] == "East":
                eastCell = (pacman[0]+1, pacman[1])
                expectedUtilityOfMoveIndex = self.coordinatesOfEachEXField.index(eastCell)
                possibleExpectedUtilityMoves.append(self.expectedUtilities[expectedUtilityOfMoveIndex])
                moveIndexes.append("East")
            elif legal[i] == "North":
                northCell = (pacman[0], pacman[1]+1)
                expectedUtilityOfMoveIndex = self.coordinatesOfEachEXField.index(northCell)
                possibleExpectedUtilityMoves.append(self.expectedUtilities[expectedUtilityOfMoveIndex])
                moveIndexes.append("North")
            elif legal[i] == "South":
                southCell = (pacman[0], pacman[1]-1)
                expectedUtilityOfMoveIndex = self.coordinatesOfEachEXField.index(southCell)
                possibleExpectedUtilityMoves.append(self.expectedUtilities[expectedUtilityOfMoveIndex])
                moveIndexes.append("South")
        
        maxUtilityMoveIndex = possibleExpectedUtilityMoves.index(max(possibleExpectedUtilityMoves)) # index of maximum utility is stored 
        maxUtilityMove = moveIndexes[maxUtilityMoveIndex] # the move that corresponds to that utility is picked and stored in maxUtilityMove variable
        if maxUtilityMove in legal :
            return api.makeMove(maxUtilityMove, legal) # maxUtilityMove is carried out by pacman

