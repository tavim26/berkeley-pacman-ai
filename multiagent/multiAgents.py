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
    """

    def getAction(self, gameState: GameState):
        """
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Evaluates a state-action pair: the higher the value, the better the
        action. The score of the successor state already rewards eating food
        (+10) and penalizes wasted time (-1 per move), so the terms below only
        add guidance towards food and away from dangerous ghosts.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        score = successorGameState.getScore()

        # Move towards the closest food
        foodList = newFood.asList()
        if foodList:
            closestFood = min(manhattanDistance(newPos, food) for food in foodList)
            score += 1.0 / closestFood if closestFood > 0 else 1.0

        for ghost in newGhostStates:
            distance = manhattanDistance(newPos, ghost.getPosition())
            if ghost.scaredTimer > distance:
                # Scared ghost that can be reached before it recovers: chase it
                score += 2.0 / (distance + 1)
            elif ghost.scaredTimer == 0:
                # Active ghost: heavily penalize getting close to it
                if distance <= 1:
                    score -= 500
                elif distance <= 3:
                    score -= 5.0 / distance

        # Standing still wastes time
        if action == Directions.STOP:
            score -= 2

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

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent. A search "depth" is one move of Pacman followed by one
    move of every ghost. Pacman maximizes, ghosts minimize.
    """

    def getAction(self, gameState: GameState):
        numAgents = gameState.getNumAgents()

        def value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            values = [value(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
                      for action in actions]
            return max(values) if agentIndex == 0 else min(values)

        nextAgent = 1 % numAgents
        nextDepth = 1 if nextAgent == 0 else 0

        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            score = value(gameState.generateSuccessor(0, action), nextAgent, nextDepth)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning. Pruning is strict (a branch is cut
    only when its value is strictly worse than the bound), and no pruning is
    done at the root, so the chosen action is the same as with plain minimax.
    """

    def getAction(self, gameState: GameState):
        numAgents = gameState.getNumAgents()

        def value(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:  # Pacman: maximize
                v = float('-inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = max(v, value(successor, nextAgent, nextDepth, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            else:  # Ghost: minimize
                v = float('inf')
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    v = min(v, value(successor, nextAgent, nextDepth, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        nextAgent = 1 % numAgents
        nextDepth = 1 if nextAgent == 0 else 0

        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = value(successor, nextAgent, nextDepth, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent. Pacman maximizes; every ghost is modeled as choosing
    uniformly at random among its legal moves, so ghost nodes return the
    average of their children.
    """

    def getAction(self, gameState: GameState):
        numAgents = gameState.getNumAgents()

        def value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            values = [value(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
                      for action in actions]
            if agentIndex == 0:
                return max(values)
            return sum(values) / float(len(values))

        nextAgent = 1 % numAgents
        nextDepth = 1 if nextAgent == 0 else 0

        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            score = value(gameState.generateSuccessor(0, action), nextAgent, nextDepth)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Evaluation function for the adversarial agents.

    DESCRIPTION:
    Starts from the game score and adds guidance terms:
      - fewer food dots remaining is better (-4 per dot);
      - closer to the nearest food dot is better (-1 per step);
      - fewer capsules remaining is better (-20 per capsule);
      - active ghosts within distance 3 are penalized, more the closer they are;
      - scared ghosts that can still be reached are attractive, more the closer
        they are (eating one gives +200 through the game score).
    Terminal states return the game score directly (a win includes +500 and a
    loss includes -500).
    """
    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()

    position = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    score -= 4 * len(foodList)
    if foodList:
        score -= min(manhattanDistance(position, food) for food in foodList)

    score -= 20 * len(capsules)

    for ghost in ghostStates:
        distance = manhattanDistance(position, ghost.getPosition())
        if ghost.scaredTimer > distance:
            score += 60.0 / (distance + 1)
        elif ghost.scaredTimer == 0 and distance <= 3:
            score -= (4 - distance) * 25

    return score


# Abbreviation
better = betterEvaluationFunction