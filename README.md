# Pacman AI: Search and Adversarial Agents

University project implementing classic Artificial Intelligence algorithms in the Pacman environment, based on the [UC Berkeley CS188 Pacman projects](http://ai.berkeley.edu).

The repository contains two projects:

| Project | Topic | Folder |
|---|---|---|
| Project 1 | Uninformed and informed search | [`search/`](search/) |
| Project 2 | Multi-agent and adversarial search | [`multiagent/`](multiagent/) |

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Project 1: Search](#project-1-search)
  - [Project 2: Multi-Agent](#project-2-multi-agent)
- [Testing](#testing)
- [Implementation Notes](#implementation-notes)
- [Results](#results)
- [Credits and License](#credits-and-license)

---

## Features

### Project 1: Search

- **Depth-First Search (DFS)**
- **Breadth-First Search (BFS)**
- **Uniform Cost Search (UCS)**
- **A\* Search** with pluggable heuristics
- **Corners Problem**: shortest path that visits all four corners of the maze, with an admissible and consistent heuristic
- **Food Search Problem**: eating all dots using A\* and a heuristic based on real maze distances
- **Closest Dot Agent**: greedy, suboptimal strategy that repeatedly goes to the nearest dot

### Project 2: Multi-Agent

- **Reflex Agent** with a hand-crafted state evaluation function
- **Minimax Agent** (Pacman maximizes, ghosts minimize)
- **Alpha-Beta Pruning Agent**
- **Expectimax Agent** (ghosts modeled as uniformly random)
- **Custom evaluation function** (`betterEvaluationFunction`)

---

## Repository Structure

```
.
├── README.md
├── search/                 # Project 1
│   ├── search.py           # DFS, BFS, UCS, A*
│   ├── searchAgents.py     # Corners, Food, ClosestDot problems and heuristics
│   ├── pacman.py           # Game engine (provided)
│   ├── layouts/            # Maze layouts
│   ├── test_cases/         # Autograder test cases
│   └── ...
└── multiagent/             # Project 2
    ├── multiAgents.py      # Reflex, Minimax, Alpha-Beta, Expectimax, evaluation
    ├── pacman.py           # Game engine (provided)
    ├── layouts/            # Game layouts
    ├── test_cases/         # Autograder test cases
    └── ...
```

The algorithms are implemented in `search/search.py`, `search/searchAgents.py` and `multiagent/multiAgents.py`. The remaining files are the provided framework (game engine, graphics, autograder).

---

## Requirements

- Python 3.8 or newer
- Tkinter (for the graphical interface)

On Debian/Ubuntu/Linux Mint:

```bash
sudo apt install python3 python3-tk
```

No external Python packages are needed.

---

## Usage

Clone the repository:

```bash
git clone https://github.com/tavim26/berkeley-pacman-ai.git
cd berkeley-pacman-ai
```

### Project 1: Search

Run all commands from the `search/` folder.

```bash
cd search

# Uninformed search
python3 pacman.py -l tinyMaze -p SearchAgent -a fn=dfs
python3 pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python3 pacman.py -l mediumMaze -p SearchAgent -a fn=ucs

# A* with the Manhattan heuristic
python3 pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

# Corners problem
python3 pacman.py -l mediumCorners -p AStarCornersAgent

# Eating all the dots
python3 pacman.py -l trickySearch -p AStarFoodSearchAgent

# Suboptimal search (closest dot first)
python3 pacman.py -l bigSearch -p ClosestDotSearchAgent
```

### Project 2: Multi-Agent

Run all commands from the `multiagent/` folder.

```bash
cd multiagent

# Reflex agent
python3 pacman.py -p ReflexAgent -l testClassic
python3 pacman.py --frameTime 0 -p ReflexAgent -k 2

# Minimax and Alpha-Beta
python3 pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
python3 pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3

# Expectimax
python3 pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3

# Custom evaluation function, 10 games without graphics
python3 pacman.py -p ExpectimaxAgent -l smallClassic -a evalFn=better -q -n 10
```

Useful options: `-l` layout, `-p` agent, `-a` agent arguments, `-k` number of ghosts, `-n` number of games, `-q` no graphics, `-z` zoom, `--frameTime` delay between frames.

---

## Testing

Both projects include the Berkeley autograder. Run it from inside each project folder:

```bash
python3 autograder.py --no-graphics        # all questions
python3 autograder.py -q q3 --no-graphics  # a single question
```

| Project | Autograder result |
|---|---|
| `search/` | 26/25 (all questions passed, including the extra credit on the food heuristic) |
| `multiagent/` | All tests passed |

---

## Implementation Notes

### Search algorithms

- All algorithms use **graph search**: a set of visited states prevents expanding the same state twice.
- DFS uses `util.Stack`, BFS uses `util.Queue`, and UCS and A\* use `util.PriorityQueue`.
- UCS and A\* re-expand a state only when a strictly cheaper path to it is found.

### Corners problem

- **State:** `(position, visitedCorners)`, where `visitedCorners` is a sorted tuple, so equivalent states always have the same representation.
- **Heuristic:** length of the shortest tour, using Manhattan distances, that visits all unvisited corners, found by checking every visiting order (at most 4! = 24). Manhattan distance never overestimates the real maze distance, so the heuristic is **admissible**.

### Food search

- **Heuristic:** real maze distance (computed with BFS and cached) from Pacman to the farthest remaining dot. Pacman must at least reach that dot, so the heuristic is **admissible and consistent**.

### Adversarial agents

- A search *depth* is one Pacman move followed by one move of every ghost.
- **Alpha-Beta** returns the same action as Minimax while expanding fewer states, because pruning is strict and not applied at the root.
- **Expectimax** replaces the ghosts' `min` nodes with the average over their legal moves.
- **`betterEvaluationFunction`** starts from the game score and adds terms for remaining food, distance to the nearest dot, remaining capsules, proximity to active ghosts (penalty) and to scared ghosts (reward).

---


## Credits and License

The Pacman AI projects were developed at **UC Berkeley** for the course CS188: Introduction to Artificial Intelligence. The core projects and autograders were primarily created by John DeNero and Dan Klein. Student-side autograding was added by Brad Miller, Nick Hay and Pieter Abbeel. See <http://ai.berkeley.edu>.

The framework files keep their original Berkeley licensing notice. This repository is shared **for educational purposes only**. If you are taking a course based on these projects, do not copy this code: follow your institution's academic integrity rules.
