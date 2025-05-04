#PART 1 Maze Solver
# This program implements a maze solver using A* and Greedy search algorithms.
# It reads a maze from a file, finds the path from start to goal, and visualizes the result.
# The maze is represented as a grid with walls, start point 'A', and goal point 'B'.
# The program uses a priority queue to explore the most promising nodes first.
# It also provides a visualization of the explored nodes and the solution path.

#Import necessary libraries
import os
import heapq
import matplotlib.pyplot as plt
import numpy as np

#Node class to represent states in the maze
class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state      # (row, col)
        self.parent = parent    # parent node
        self.action = action    # Action to reach this node
        self.cost = cost        # g(n): cost from start node

    def __lt__(self, other):
        return False # For priority queue comparison

#Maze class for parsing and logic
class Maze:
    def __init__(self, filename):
        self.read_maze(filename)
        self.explored = set()
        self.solution_path = []

    def read_maze(self, filename):
        with open(filename) as f:
            lines = f.read().splitlines()
        self.height = len(lines)
        self.width = max(len(line) for line in lines)
        self.walls = set()

        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char == "A":
                    self.start = (i, j)
                elif char == "B":
                    self.goal = (i, j)
                elif char == "#":
                    self.walls.add((i, j))

    def neighbors(self, state): #Get neighbors for the current state
        i, j = state   
        candidates = [(i-1, j, "up"), (i+1, j, "down"), 
                      (i, j-1, "left"), (i, j+1, "right")]
        results = []
        for x, y, action in candidates:
            if 0 <= x < self.height and 0 <= y < self.width and (x, y) not in self.walls:
                results.append(((x, y), action))
        return results

    def heuristic(self, state):
        (x1, y1) = state
        (x2, y2) = self.goal
        return abs(x1 - x2) + abs(y1 - y2)

    def reconstruct_path(self, node):
        path = []
        while node.parent is not None:
            path.append(node.state)
            node = node.parent
        path.reverse()
        self.solution_path = path

    def solve(self, algorithm="greedy"):
        start_node = Node(state=self.start, cost=0)
        frontier = []
        heapq.heappush(frontier, (self.heuristic(self.start), start_node))
        self.explored = set()

        while frontier:
            _, current = heapq.heappop(frontier)
            if current.state == self.goal:
                self.reconstruct_path(current)
                return self.solution_path

            self.explored.add(current.state)

            for neighbor, action in self.neighbors(current.state):
                if neighbor in self.explored:
                    continue
                cost = current.cost + 1
                new_node = Node(state=neighbor, parent=current, action=action, cost=cost)

                if algorithm == "greedy":
                    priority = self.heuristic(neighbor)
                elif algorithm == "astar":
                    priority = cost + self.heuristic(neighbor)
                else:
                    raise ValueError("Unsupported algorithm")

                heapq.heappush(frontier, (priority, new_node))

        raise Exception("No path found")

    def visualize(self, output_file):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255

        for (i, j) in self.walls:
            img[i, j] = [0, 0, 0]

        for (i, j) in self.explored:
            img[i, j] = [192, 192, 192]

        for (i, j) in self.solution_path:
            img[i, j] = [0, 255, 0]

        si, sj = self.start
        gi, gj = self.goal
        img[si, sj] = [0, 0, 255]
        img[gi, gj] = [255, 0, 0]

        plt.imshow(img)
        plt.axis("off")
        plt.savefig(output_file)
        plt.close()
