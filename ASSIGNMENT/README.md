# Artificial Intelligence Assignment 1

**Course:** ARI711S - Artificial Intelligence
**Group Name:** CTRL +ALT+ELITE

## Overview
This repository contains solutions for 4 AI problems:
1. Maze path-finding with informed search
2. Traveling Salesman Problem with Simulated Annealing
3. Optimal Tic-Tac-Toe AI
4. GridWorld Q-learning

## Dependencies & Requirements
- Python 3.8+
- numpy>=1.21.0
- matplotlib>=3.5.0

----------------------------------------------------------------------------

# QUESTION 1 - Maze Path-finding with Informed Search

## Description
Implementation of Greedy Best-First Search and A* Search algorithms to find the shortest path in a maze.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python maze.py`

## Output
- Prints the path found by each algorithm
- Generates visualization images:
  - `greedy_solution.png`
  - `astar_solution.png`

## Custom Mazes
Edit `maze.txt` to create different mazes:
- 'A' = Start
- 'B' = Goal
- '#' = Wall
- '.' = Empty space

----------------------------------------------------------------------------

# QUESTION 2 - Traveling Salesman Problem with Simulated Annealing

## Description
Solution to the TSP problem for 10 Namibian towns using Simulated Annealing.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python tsp.py`

## Output
- Prints initial and optimized route distances
- Generates:
  - `convergence.png`: Shows optimization progress
  - `route.png`: Visualizes the best route found

## Customisation
Edit the distance matrix in `tsp.py` to solve different TSP instances.

----------------------------------------------------------------------------

# QUESTION 3 - Unbeatable Tic-Tac-Toe AI

## Description
Implementation of Minimax algorithm for optimal Tic-Tac-Toe play.

## How to Run
```bash`
python runner.py

----------------------------------------------------------------------------

# QUESTION 4 - GridWorld Q-learning

## Description
Implementation of Q-learning to find optimal policy for a GridWorld MDP.

## How to Run
```bash
python gridworld.py
