# Importing required libraries
import heapq
from visualization import visualize
import os

class Maze:
    def __init__(self, filename):
        self.grid = []
        self.start = None
        self.goal = None
        self.walls = []
        self.parse_maze(filename)

    def parse_maze(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Maze file '{filename}' not found.")
            
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                row = []
                for j, char in enumerate(line.strip()):
                    if char == 'A':
                        self.start = (i, j)
                        row.append(0)
                    elif char == 'B':
                        self.goal = (i, j)
                        row.append(0)
                    elif char == '#':
                        self.walls.append((i, j))
                        row.append(1)
                    else:
                        row.append(0)
                self.grid.append(row)

    def neighbors(self, state):
        row, col = state
        directions = [
            ("up", (row-1, col)),
            ("down", (row+1, col)),
            ("left", (row, col-1)),
            ("right", (row, col+1))
        ]
        return [
            (action, (r, c)) for action, (r, c) in directions
            if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]) and self.grid[r][c] != 1
        ]

class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
    
    # Add comparison methods to make Nodes comparable
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __eq__(self, other):
        return self.cost == other.cost

def solve(maze, algorithm="greedy"):
    frontier = []
    # Push a tuple of (priority, unique_id, node) to avoid comparing Nodes
    unique_id = 0
    heapq.heappush(frontier, (0, unique_id, Node(maze.start)))
    unique_id += 1
    explored = set()
    explored_states = []

    while frontier:
        _, _, node = heapq.heappop(frontier)

        if node.state == maze.goal:
            path = []
            while node.parent:
                path.append(node.action)
                node = node.parent
            return path[::-1], explored_states

        if node.state in explored:
            continue

        explored.add(node.state)
        explored_states.append(node.state)

        for action, state in maze.neighbors(node.state):
            if state not in explored:
                if algorithm == "greedy":
                    priority = abs(state[0] - maze.goal[0]) + abs(state[1] - maze.goal[1])
                elif algorithm == "astar":
                    priority = node.cost + 1 + abs(state[0] - maze.goal[0]) + abs(state[1] - maze.goal[1])
                
                heapq.heappush(frontier, (priority, unique_id, Node(state, node, action, node.cost + 1)))
                unique_id += 1

    raise Exception("No path exists from start to goal")

def create_sample_maze(filename="maze.txt"):
    sample_maze = """\
# A # # # # #
# 0 0 0 0 0 #
# # # 0 # # #
# 0 0 0 0 # #
# 0 # # # # #
# 0 0 0 0 B #
# # # # # # #
"""
    with open(filename, 'w') as f:
        f.write(sample_maze)

if __name__ == "__main__":
    maze_file = "maze.txt"
    
    if not os.path.exists(maze_file):
        print(f"Creating sample maze file '{maze_file}'...")
        create_sample_maze(maze_file)

    try:
        maze = Maze(maze_file)
        
        print("\nSolving with Greedy Best-First Search:")
        try:
            path, explored = solve(maze, "greedy")
            print("Solution Path:", path)
            visualize(maze, path, explored, "greedy_solution.png")
        except Exception as e:
            print(f"Greedy search failed: {e}")

        print("\nSolving with A* Search:")
        try:
            path, explored = solve(maze, "astar")
            print("Solution Path:", path)
            visualize(maze, path, explored, "astar_solution.png")
        except Exception as e:
            print(f"A* search failed: {e}")
            
    except Exception as e:
        print(f"\nError: {e}")