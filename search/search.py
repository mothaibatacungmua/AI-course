# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    closed = []
    seq = []
    stack = util.Stack()
    startState = problem.getStartState()
    startSuccessors = problem.getSuccessors(startState)

    for s in startSuccessors:
        stack.push(s)

    def isVisited(state):
        try:
            closed.index("%d-%d" % (state[0][0],state[0][1]))
            return True
        except ValueError:
            return False

    def insertVisiting(state):
        closed.append("%d-%d" % (state[0][0],state[0][1]))

    def isBackward(successors):
        if len(successors) == 0:
            return True

        for s in successors:
            if not isVisited(s):
                return False
        return True

    closed.append("%d-%d" % (startState[0], startState[1]))

    while not stack.isEmpty():
        frontElem = stack.pop()
        
        #check is goal state, return sequence of move
        if problem.isGoalState(frontElem[0]):
            seq.append(frontElem)
            moves = []
            for s in seq:
                moves.append(s[1])
            return moves

        if not isVisited(frontElem):
            insertVisiting(frontElem)
            seq.append(frontElem)

            successors = problem.getSuccessors(frontElem[0])

            if isBackward(successors):
                while True:
                    if len(seq) == 0:
                        break
                    p = seq.pop()
                    successors = problem.getSuccessors(p[0])
                    if not isBackward(successors):
                        #restore parent
                        seq.append(p)
                        break
                    pass
            else:
                for s in successors:
                    if not isVisited(s):
                        stack.push(s)

    return None

    #util.raiseNotDefined()

from game import Directions
from game import Actions

def isVisited(state, closed):
    for v in closed:
        if (state[0] == v[0][0]) and (state[1] == v[0][1]):
            return v
    return None
    pass

def insertVisiting(state, closed):
    closed.append(state)
    pass

def getPathFromParent(startState, state, closed, reverse=False):
    moves = []
    moves.append(state)
    s = state
    
    while True:
        reverseState = Actions.getSuccessor(s[0], Actions.reverseDirection(s[1]))

        realParent = isVisited(reverseState, closed)
        if (realParent != None) and (realParent[0] != startState):
            moves.append((realParent[0], realParent[1]))
            s = realParent
        else:
            break
        pass

    moves.append((startState, None))
    if not reverse:
        moves.reverse()
    return moves

def traceBack(startState, nodes, closed):
    seq = []

    if len(nodes) == 1:
        seq = getPathFromParent(startState, nodes[0], closed, False)
        return seq

    def findParentBetweenTwoNodes(pathA, pathB):
        for i in range(len(pathA)-1, -1, -1):
            for j in range(len(pathB)-1, -1, -1):
                if (pathA[i][0][0] == pathB[j][0][0]) and (pathA[i][0][1] == pathB[j][0][1]):
                    return (i, j)
        return (None, None)

    prevPath = getPathFromParent(startState, nodes[len(nodes)-1], closed, True)

    for i in range(len(nodes)-2, -1, -1):
        nextPath = getPathFromParent(startState, nodes[i], closed, True)
        x, y = findParentBetweenTwoNodes(prevPath, nextPath)

        
        #add trace back 1
        seq.extend(prevPath[:x])

        #add trace back 2
        for j in range(y-1, -1, -1):
            seq.append((\
                Actions.getSuccessor(nextPath[j][0], Actions.reverseDirection(nextPath[j][1])), \
                Actions.reverseDirection(nextPath[j][1]) \
            ))

        #add trace back 3
        seq.extend(nextPath[:y])

        prevPath = nextPath[y:len(nextPath)]

    seq.append((startState, None))
    seq.reverse()

    return seq

def unpackSeq(rawMoves):
    unpackMoves = []
    for i in range(1, len(rawMoves)):
        unpackMoves.append(rawMoves[i][1])  

    return unpackMoves

def breadthFirstSearchV1(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed = []
    queue = util.Queue()
    startState = problem.getStartState()
    startSuccessors = problem.getSuccessors(startState)

    for s in startSuccessors:
        queue.push(s)

    subtargets = []
    
    insertVisiting((startState,None), closed)

    while not queue.isEmpty():
        frontElem = queue.pop()

        #check is goal state, return sequence of move
        if problem.isGoalState(frontElem[0]):
            insertVisiting((frontElem[0], frontElem[1]), closed)
            #trace back for find path from root
            subtargets.append(frontElem)
            break
            
        if hasattr(problem.__class__, 'isSubtarget') and callable(getattr(problem.__class__, 'isSubtarget')):
            if problem.isSubtarget(frontElem[0]) and isVisited(frontElem[0], closed) == None:
                subtargets.append(frontElem)

        if isVisited(frontElem[0], closed) == None:
            insertVisiting((frontElem[0], frontElem[1]), closed)

            successors = problem.getSuccessors(frontElem[0])
            
            for s in successors:
                if isVisited(s[0], closed) == None:
                    queue.push(s)

    moves = unpackSeq(traceBack(startState, subtargets, closed))
    return moves


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed = []
    queue = util.Queue()
    startState = problem.getStartState()
    initState = startState
    startSuccessors = problem.getSuccessors(startState)

    for s in startSuccessors:
        queue.push(s)

    insertVisiting((startState,None), closed)
    seq = []

    while not queue.isEmpty():
        frontElem = queue.pop()

        
            
        if hasattr(problem.__class__, 'isSubtarget') and callable(getattr(problem.__class__, 'isSubtarget')):
            if problem.isSubtarget(frontElem[0]) and not problem.isGoalState(frontElem[0]):

                seq.extend(traceBack(startState, [frontElem], closed)[1:])
                startState = frontElem[0]
                closed = []
                queue = util.Queue()
                startSuccessors = problem.getSuccessors(startState)

                for s in startSuccessors:
                    queue.push(s)
        
        #check is goal state, return sequence of move
        if problem.isGoalState(frontElem[0]):
            insertVisiting((frontElem[0], frontElem[1]), closed)
            #trace back for find path from root
            #subtargets.append(frontElem)
            seq.extend(traceBack(startState, [frontElem], closed)[1:])
            break        

        if isVisited(frontElem[0], closed) == None:
            insertVisiting((frontElem[0], frontElem[1]), closed)

            successors = problem.getSuccessors(frontElem[0])
            
            for s in successors:
                if isVisited(s[0], closed) == None:
                    queue.push(s)

    seq.insert(0, (initState,None))
    moves = unpackSeq(seq)
    return moves

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    closed = []

    def getCost(item):
        return item[2]

    priorityQueue = util.PriorityQueueWithFunction(getCost)

    startState = problem.getStartState()
    startSuccessors = problem.getSuccessors(startState)

    for s in startSuccessors:
        priorityQueue.push(s)

    subtargets = []
    insertVisiting((startState,None), closed)
    while not priorityQueue.isEmpty():
        frontElem = priorityQueue.pop()
        
        #check is goal state, return sequence of move
        if problem.isGoalState(frontElem[0]):
            insertVisiting((frontElem[0], frontElem[1]), closed)
            
            subtargets.append(frontElem)
            break

        if hasattr(problem.__class__, 'isSubtarget') and callable(getattr(problem.__class__, 'isSubtarget')):
            if problem.isSubtarget(frontElem[0]):
                subtargets.append(frontElem)

        if isVisited(frontElem, closed) == None:
            insertVisiting((frontElem[0], frontElem[1]), closed)

            successors = problem.getSuccessors(frontElem[0])
            
            for s in successors:
                if isVisited(s, closed) == None:
                    priorityQueue.push((s[0], s[1], frontElem[2] + s[2]))

    moves = unpackSeq(traceBack(startState, subtargets, closed))
    return moves

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    closed = []

    def getCost(item):
        return item[2] + item[3]

    priorityQueue = util.PriorityQueueWithFunction(getCost)

    startState = problem.getStartState()
    startSuccessors = problem.getSuccessors(startState)

    for s in startSuccessors:
        priorityQueue.push((s[0], s[1], s[2], heuristic(s[0], problem)))

    subtargets = []
    seq = []

    insertVisiting((startState,None), closed)

    while not priorityQueue.isEmpty():
        frontElem = priorityQueue.pop()
        
        if hasattr(problem.__class__, 'isSubtarget') and callable(getattr(problem.__class__, 'isSubtarget')):
            if problem.isSubtarget(frontElem[0]) and not problem.isGoalState(frontElem[0]):

                print 'seq extend'
                seq.extend(traceBack(startState, [frontElem], closed)[1:])

                startState = frontElem[0]
                closed = []
                queue = util.PriorityQueueWithFunction(getCost)
                startSuccessors = problem.getSuccessors(startState)

                for s in startSuccessors:
                    queue.push((s[0], s[1], s[2], heuristic(s[0], problem)))

        #check is goal state, return sequence of move
        if problem.isGoalState(frontElem[0]):
            insertVisiting((frontElem[0], frontElem[1]), closed)
            #trace back for find path from root
            seq.extend(traceBack(startState, [frontElem], closed)[1:])
            break

        if isVisited(frontElem[0], closed) == None:
            insertVisiting((frontElem[0], frontElem[1]), closed)

            successors = problem.getSuccessors(frontElem[0])
            
            for s in successors:
                if isVisited(s[0], closed) == None:
                    priorityQueue.push((\
                        s[0], s[1], \
                        frontElem[2] + s[2],\
                        heuristic(s[0], problem)\
                        ))

    seq.insert(0, (initState,None))
    moves = unpackSeq(seq)
    return moves


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
