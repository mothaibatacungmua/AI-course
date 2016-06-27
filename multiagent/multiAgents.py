# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import sys

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def __init__(self):
      self.posSaved = {}

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        currentPos = gameState.getPacmanPosition()
        self.setPositionReward(currentPos)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        oldFood = currentGameState.getFood()


        return self.calcKNearestFoodReward(3, newPos, oldFood, newFood, successorGameState.getCapsules(), successorGameState.getWalls()) + \
               self.calcGhostReward(newGhostStates, newPos) + \
               self.calcScaredReward(newGhostStates, newPos) + \
               self.getPositionReward(newPos)

    FOOD_REWARD = 2
    CAPSULE_REWARD = 4
    GHOST_REWARD = -5
    DECAY = 0.8
    OLD_POS_REWARD = -0.5

    def getPositionReward(self, pos):
      if self.posSaved.has_key(pos):
         return self.posSaved[pos]

      return 0

    def setPositionReward(self, pos):
      if self.posSaved.has_key(pos):
         self.posSaved[pos] = self.posSaved[pos] + ReflexAgent.OLD_POS_REWARD
         return self.posSaved[pos]

      self.posSaved[pos] = 0

      return 0
      

    def calcKNearestFoodReward(self, k, pacmanPos, oldFoodGrid, newFoodGrid, capsulesList, walls):
      closed = []
      queue = util.Queue()

      queue.push(pacmanPos)
      closed.append(pacmanPos)

      findFood = 0
      totalScore = 0.0

      count = 0

      while not queue.isEmpty():
        travel = queue.pop()
        count = count + 1
        #print count
        #print travel
        if newFoodGrid[travel[0]][travel[1]]:
          findFood = findFood + 1

          if travel in capsulesList:
            totalScore = totalScore + pow(ReflexAgent.DECAY,manhattanDistance(pacmanPos, travel)) * ReflexAgent.CAPSULE_REWARD
          else:
            totalScore = totalScore + pow(ReflexAgent.DECAY,manhattanDistance(pacmanPos, travel)) * ReflexAgent.FOOD_REWARD


        if findFood == k:
          break

        for i in [-1, 0, 1]:
          if (travel[0] + i) < 0 or (travel[0] + i) > (newFoodGrid.width - 1):
              continue

          for j in [-1, 0, 1]:
            if (travel[1] + j) < 0 or (travel[1] + j) > (newFoodGrid.height - 1):
              continue

            newTravel = (travel[0] + i, travel[1] + j)
            if walls[travel[0] + i][travel[1] + j] or (newTravel in closed):
              continue

            closed.append(newTravel)
            queue.push(newTravel)

      if findFood == 0:
        score = 0
      else:
        score = 1.0/findFood * totalScore

      if oldFoodGrid[pacmanPos[0]][pacmanPos[1]]:
        score = score + ReflexAgent.FOOD_REWARD

      #print '--\n'
      #print pacmanPos
      #print 'Food score:%0.3f' % score
      return score

    def calcGhostReward(self, ghostStates, pacmanPos):
      numGhosts = len(ghostStates)
      totalScore = 0.0
      for ghostState in ghostStates:
        totalScore = totalScore + (ghostState.scaredTimer == 0) *(\
                     ReflexAgent.GHOST_REWARD * \
                     (manhattanDistance(pacmanPos, ghostState.getPosition()) == 1) +
                     2*ReflexAgent.GHOST_REWARD * \
                     (manhattanDistance(pacmanPos, ghostState.getPosition()) == 0))

      #print 'Ghost reward:%0.3f' % totalScore

      return totalScore

    def calcScaredReward(self, ghostStates, pacmanPos):
      totalScore = 0.0
      for ghostState in ghostStates:
        totalScore = totalScore + (ghostState.scaredTimer > 0)*(ghostState.scaredTimer - manhattanDistance(pacmanPos, ghostState.getPosition()))

      return totalScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    FOOD_REWARD = 2
    CAPSULE_REWARD = 4
    GHOST_REWARD = -5
    DECAY = 0.8
    OLD_POS_REWARD = -0.5

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def __init__(self, *args, **kwargs):
      self.posSaved = {}
      MultiAgentSearchAgent.__init__(self, *args, **kwargs)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        pacmanIndex = 0
        ghostIndex = 1
        numberAgents = gameState.getNumAgents()
        maxGhostIndex = numberAgents - 1
        
        maxScore, action = self.maxNode(gameState, 1, pacmanIndex, maxGhostIndex)

        return action
        #util.raiseNotDefined()


    def maxNode(self, gameState, depth, pacmanIndex, maxGhostIndex):
      if depth > self.depth:
        return (self.evaluationFunction(gameState), None)

      legalMoves = gameState.getLegalActions(pacmanIndex)
      if len(legalMoves) == 0:
        return (self.evaluationFunction(gameState), None)
      
      calcMins = []
      firstGhostIndex = 1

      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(pacmanIndex, action)
        calcMins.append(self.minNode(successorGameState, depth, firstGhostIndex,maxGhostIndex, pacmanIndex)[0])

      maxScore = max(calcMins)
      indices = [i for i in range(len(calcMins)) if calcMins[i] == maxScore]

      return (maxScore, legalMoves[indices[0]])
      

    def minNode(self, gameState, depth, ghostIndex, maxGhostIndex, pacmanIndex):
      legalMoves = gameState.getLegalActions(ghostIndex)

      if len(legalMoves) == 0:
        return (self.evaluationFunction(gameState), None)

      calcMins = []

      if ghostIndex < maxGhostIndex:
        for action in legalMoves:
          successorGameState = gameState.generateSuccessor(ghostIndex, action)
          calcMins.append(self.minNode(successorGameState, depth, ghostIndex + 1, maxGhostIndex, pacmanIndex)[0])

      if ghostIndex == maxGhostIndex:
        for action in legalMoves:
          successorGameState = gameState.generateSuccessor(ghostIndex, action)
          calcMins.append(self.maxNode(successorGameState, depth + 1, pacmanIndex, maxGhostIndex)[0])

      minScore = min(calcMins)
      indices = [i for i in range(len(calcMins)) if calcMins[i] == minScore]

      return (minScore, legalMoves[indices[0]])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def __init__(self, *args, **kwargs):
      self.posSaved = {}
      MultiAgentSearchAgent.__init__(self, *args, **kwargs)


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        pacmanIndex = 0
        ghostIndex = 1
        numberAgents = gameState.getNumAgents()
        maxGhostIndex = numberAgents - 1

        alpha = -sys.maxint
        beta = sys.maxint

        maxScore, action = self.maxNode(alpha, beta, gameState, 1, pacmanIndex, maxGhostIndex)

        return action

    def maxNode(self, alpha, beta, gameState, depth, pacmanIndex, maxGhostIndex):
      if depth > self.depth:
        return (self.evaluationFunction(gameState), None)

      legalMoves = gameState.getLegalActions(pacmanIndex)
      if len(legalMoves) == 0:
        return (self.evaluationFunction(gameState), None)
      
      firstGhostIndex = 1
      v = -sys.maxint * 1.0
      track = None

      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(pacmanIndex, action)
        t = v
        v = max(v, self.minNode(alpha, beta, successorGameState, depth, firstGhostIndex,maxGhostIndex, pacmanIndex)[0])

        if t != v:
          track = action
        #prunning
        if (v > beta):
          return (v, action)

        alpha = max(alpha, v)

      return (v, track)
      

    def minNode(self, alpha, beta, gameState, depth, ghostIndex, maxGhostIndex, pacmanIndex):
      legalMoves = gameState.getLegalActions(ghostIndex)

      if len(legalMoves) == 0:
        return (self.evaluationFunction(gameState), None)
        
      calcMins = []
      v = sys.maxint * 1.0
      track = None

      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(ghostIndex, action)
        t = v
        if ghostIndex < maxGhostIndex:
          v = min(v, self.minNode(alpha, beta, successorGameState, depth, ghostIndex + 1, maxGhostIndex, pacmanIndex)[0])
        else:  
          v = min(v, self.maxNode(alpha, beta, successorGameState, depth + 1, pacmanIndex, maxGhostIndex)[0])

        if t != v:
          track = action

        #prunning
        if (v < alpha):
          return (v, action)
        
        beta = min(beta, v)

      return (v, track)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def __init__(self, *args, **kwargs):
      self.posSaved = {}
      MultiAgentSearchAgent.__init__(self, *args, **kwargs)


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        pacmanIndex = 0
        ghostIndex = 1
        numberAgents = gameState.getNumAgents()
        maxGhostIndex = numberAgents - 1
        
        maxScore, action = self.maxNode(gameState, 1, pacmanIndex, maxGhostIndex)

        return action

    def maxNode(self, gameState, depth, pacmanIndex, maxGhostIndex):
      if depth > self.depth:
        return (self.evaluationFunction(gameState), None)

      legalMoves = gameState.getLegalActions(pacmanIndex)
      if len(legalMoves) == 0:
        return (self.evaluationFunction(gameState), None)
      
      calcMins = []
      firstGhostIndex = 1

      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(pacmanIndex, action)
        calcMins.append(self.expectNode(successorGameState, depth, firstGhostIndex,maxGhostIndex, pacmanIndex)[0])

      maxScore = max(calcMins)
      indices = [i for i in range(len(calcMins)) if calcMins[i] == maxScore]

      return (maxScore, legalMoves[random.choice(indices)])

    def expectNode(self, gameState, depth, ghostIndex, maxGhostIndex, pacmanIndex):
      legalMoves = gameState.getLegalActions(ghostIndex)
      if len(legalMoves) == 0:
        return (self.evaluationFunction(gameState), None)

      totalScore = 0.0
      for action in legalMoves:
        successorGameState = gameState.generateSuccessor(ghostIndex, action)
        if ghostIndex < maxGhostIndex:
          totalScore = totalScore + self.expectNode(successorGameState, depth, ghostIndex + 1, maxGhostIndex, pacmanIndex)[0]
        else:  
          totalScore = totalScore + self.maxNode(successorGameState, depth + 1, pacmanIndex, maxGhostIndex)[0]

      return (totalScore/len(legalMoves), None)


FOOD_REWARD = 2
CAPSULE_REWARD = 4
GHOST_REWARD = -5
DECAY = 0.8
CLEAR_REWARD = 20

def calcKNearestFoodReward(k, pacmanPos, foodGrid, capsulesList, walls):
  closed = []
  queue = util.Queue()

  queue.push(pacmanPos)
  closed.append(pacmanPos)

  findFood = 0
  totalScore = 0.0

  count = 0

  while not queue.isEmpty():
    travel = queue.pop()
    count = count + 1
    #print count
    #print travel
    if foodGrid[travel[0]][travel[1]]:
      findFood = findFood + 1

      if travel in capsulesList:
        totalScore = totalScore + pow(DECAY,manhattanDistance(pacmanPos, travel)) * CAPSULE_REWARD
      else:
        totalScore = totalScore + pow(DECAY,manhattanDistance(pacmanPos, travel)) * FOOD_REWARD


    if findFood == k:
      break

    for i in [-1, 0, 1]:
      if (travel[0] + i) < 0 or (travel[0] + i) > (foodGrid.width - 1):
          continue

      for j in [-1, 0, 1]:
        if (travel[1] + j) < 0 or (travel[1] + j) > (foodGrid.height - 1):
          continue

        newTravel = (travel[0] + i, travel[1] + j)
        if walls[travel[0] + i][travel[1] + j] or (newTravel in closed):
          continue

        closed.append(newTravel)
        queue.push(newTravel)

  if findFood == 0:
    score = 0
  else:
    score = 1.0/findFood * totalScore

  return score

def calcGhostReward(ghostStates, pacmanPos):
  numGhosts = len(ghostStates)
  totalScore = 0.0
  for ghostState in ghostStates:
    totalScore = totalScore + (ghostState.scaredTimer == 0) *(\
                 GHOST_REWARD * \
                 (manhattanDistance(pacmanPos, ghostState.getPosition()) == 1) +
                 2*GHOST_REWARD * \
                 (manhattanDistance(pacmanPos, ghostState.getPosition()) == 0))

  return totalScore

def calcScaredReward(ghostStates, pacmanPos):
  totalScore = 0.0
  for ghostState in ghostStates:
    totalScore = totalScore + (ghostState.scaredTimer > 0)*(ghostState.scaredTimer - manhattanDistance(pacmanPos, ghostState.getPosition()))

  return totalScore

def calcRemainFood(gameState):
  foodGrid = gameState.getFood()
  numFoods = gameState.getNumFood()

  score = (foodGrid.height * foodGrid.width - numFoods)

  if numFoods == 0:
    score = score + CLEAR_REWARD

  return score

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    capsulesList = currentGameState.getCapsules()
    walls = currentGameState.getWalls()    
    foodGrid = currentGameState.getFood()

    A = calcKNearestFoodReward(3, pacmanPos, foodGrid, capsulesList, walls)
    B = calcGhostReward(ghostStates, pacmanPos)
    C = calcScaredReward(ghostStates, pacmanPos)
    D = calcRemainFood(currentGameState)

    alpha = 0.2
    beta = 0.5
    gamma = 0.1
    theta = 0.2
    return  (alpha*A + beta*B  + gamma*C + theta*D)

# Abbreviation
better = betterEvaluationFunction

