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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]




    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Evaluates a state-action pair for the ReflexAgent.
        """
        # Generate the successor state for the given action
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Initialize the score with the base score of the successor state
        score = successorGameState.getScore()

        # Distance to the closest food
        foodDistances = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            score += 10 / min(foodDistances)  # Higher score for closer food

        # Ghost proximity: penalize being close to active ghosts
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostDistance = util.manhattanDistance(newPos, ghost.getPosition())
            if scaredTime == 0:  # Active ghost
                if ghostDistance > 0:  # Avoid division by zero
                    score -= 10 / ghostDistance  # Penalize proximity to active ghosts
            else:  # Scared ghost
                score += 5 / ghostDistance  # Encourage chasing scared ghosts

        # Factor in remaining food count
        score -= len(newFood.asList()) * 2  # Penalize for remaining food

        return score




def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)



class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(agentIndex, depth, state):
            # If game is over or depth is reached, evaluate state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (Max agent)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state)

            # Ghosts (Min agents)
            else:
                return minValue(agentIndex, depth, state)

        def maxValue(agentIndex, depth, state):
            actions = state.getLegalActions(agentIndex)
            if not actions:  # No legal actions
                return self.evaluationFunction(state)

            # Find the maximum value among successor states
            v = float('-inf')
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, minimax(1, depth, successor))  # Next agent is the first ghost
            return v

        def minValue(agentIndex, depth, state):
            actions = state.getLegalActions(agentIndex)
            if not actions:  # No legal actions
                return self.evaluationFunction(state)

            # Find the minimum value among successor states
            v = float('inf')
            nextAgent = (agentIndex + 1) % state.getNumAgents()  # Cycle through agents
            nextDepth = depth + 1 if nextAgent == 0 else depth  # Increment depth if Pacman's turn
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgent, nextDepth, successor))
            return v

        # Choose the best action for Pacman (agentIndex = 0)
        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)  # Start with the first ghost (agentIndex = 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(agentIndex, depth, state, alpha, beta):
            # Terminal state or depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (Max agent)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state, alpha, beta)

            # Ghosts (Min agents)
            else:
                return minValue(agentIndex, depth, state, alpha, beta)

        def maxValue(agentIndex, depth, state, alpha, beta):
            actions = state.getLegalActions(agentIndex)
            if not actions:  # No legal actions
                return self.evaluationFunction(state)

            v = float('-inf')
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, alphaBeta(1, depth, successor, alpha, beta))  # Next agent is the first ghost
                if v > beta:  # Beta cutoff
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(agentIndex, depth, state, alpha, beta):
            actions = state.getLegalActions(agentIndex)
            if not actions:  # No legal actions
                return self.evaluationFunction(state)

            v = float('inf')
            nextAgent = (agentIndex + 1) % state.getNumAgents()  # Cycle through agents
            nextDepth = depth + 1 if nextAgent == 0 else depth  # Increment depth if Pacman's turn
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, alphaBeta(nextAgent, nextDepth, successor, alpha, beta))
                if v < alpha:  # Alpha cutoff
                    return v
                beta = min(beta, v)
            return v

        # Choose the best action for Pacman (agentIndex = 0)
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = alphaBeta(1, 0, successor, alpha, beta)  # Start with the first ghost (agentIndex = 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction









class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()





def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
