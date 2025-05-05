import heapq
from visualization import visualize

class Maze:
    def __init__(self, filename):
        self.grid = []
        self.start = None
        self.goal = None
        self.walls = []
        self.parse_maze(filename)
        
    def parse_maze(self, filename):
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
        return [(action, (r, c)) for action, (r, c) in directions 
                if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]) and self.grid[r][c] != 1]

class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

def solve(maze, algorithm="greedy"):
    frontier = []
    heapq.heappush(frontier, (0, Node(maze.start)))
    explored = set()
    explored_states = []
    
    while frontier:
        _, node = heapq.heappop(frontier)
        
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
                    priority = abs(state[0]-maze.goal[0]) + abs(state[1]-maze.goal[1])
                elif algorithm == "astar":
                    priority = node.cost + 1 + abs(state[0]-maze.goal[0]) + abs(state[1]-maze.goal[1])
                heapq.heappush(frontier, (priority, Node(state, node, action, node.cost+1)))
    
    raise Exception("No path exists")

if __name__ == "__main__":
    maze = Maze("maze.txt")
    
    #greedy Best-First Search
    try:
        path, explored = solve(maze, "greedy")
        print("Greedy Path:", path)
        visualize(maze, path, explored, "greedy_solution.png")
    except Exception as e:
        print(e)
    
    #a* Search
    try:
        path, explored = solve(maze, "astar")
        print("A* Path:", path)
        visualize(maze, path, explored, "astar_solution.png")
    except Exception as e:
        print(e)