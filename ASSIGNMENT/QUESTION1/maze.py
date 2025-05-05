# Importing required libraries
import heapq  # Used to implement a priority queue for managing nodes
from visualization import visualize  # For visualizing the maze and solution paths

# ### Maze Class
class Maze:
    """
    **Represents a maze.**
    - Parses the maze from a file.
    - Tracks:
        - The start position (`A`).
        - The goal position (`B`).
        - Walls (`#`).
    - Provides methods to get valid neighboring positions.
    """
    def __init__(self, filename):
        """
        **Initialize the Maze object:**
        - Reads the maze file.
        - Sets up attributes for the grid, start, goal, and walls.
        """
        self.grid = []  # 2D list representing the maze structure
        self.start = None  # Starting position of the maze
        self.goal = None  # Goal position of the maze
        self.walls = []  # List of wall positions
        self.parse_maze(filename)  # Parse the file to populate attributes

    def parse_maze(self, filename):
        """
        **Reads the maze file and converts it into a grid.**
        - `A`: Marks the start position.
        - `B`: Marks the goal position.
        - `#`: Represents walls.
        - Open spaces are represented by `0`.
        """
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                row = []
                for j, char in enumerate(line.strip()):  # Process each character in the line
                    if char == 'A':
                        self.start = (i, j)  # Store the start position
                        row.append(0)  # Open space
                    elif char == 'B':
                        self.goal = (i, j)  # Store the goal position
                        row.append(0)  # Open space
                    elif char == '#':
                        self.walls.append((i, j))  # Add position to walls
                        row.append(1)  # Wall space
                    else:
                        row.append(0)  # Open space
                self.grid.append(row)  # Add the processed row to the grid

    def neighbors(self, state):
        """
        **Find valid neighboring positions from the current state.**
        - Valid neighbors must:
            - Be within the maze bounds.
            - Not be walls.
        - Returns a list of tuples with:
            - Direction (e.g., 'up').
            - Neighboring position as a tuple (row, col).
        """
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

# ### Node Class
class Node:
    """
    **Represents a single node in the search tree.**
    - Tracks the current position (`state`).
    - Maintains a reference to the parent node to reconstruct paths.
    - Stores the action taken to reach the node and the cumulative cost.
    """
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state  # Current position (row, col)
        self.parent = parent  # Parent node reference
        self.action = action  # Action taken to reach this node (e.g., 'up')
        self.cost = cost  # Cumulative cost from the start node to this node

# ### Solve Function
def solve(maze, algorithm="greedy"):
    """
    **Finds a path through the maze using the specified algorithm.**
    - Supported algorithms:
        1. **Greedy Best-First Search**:
            - Prioritizes nodes based on their distance to the goal (heuristic).
        2. **A* Search**:
            - Combines the path cost and the heuristic.
    - Returns:
        - `path`: A list of actions (e.g., ['right', 'down']) to reach the goal.
        - `explored_states`: A list of all visited positions (for visualization).
    """
    frontier = []  # Priority queue for nodes to explore
    heapq.heappush(frontier, (0, Node(maze.start)))  # Add the start node with priority 0
    explored = set()  # Set of explored positions
    explored_states = []  # List of all explored states for visualization

    while frontier:
        # Get the node with the lowest priority
        _, node = heapq.heappop(frontier)

        # Check if we've reached the goal
        if node.state == maze.goal:
            path = []  # To reconstruct the path
            while node.parent:  # Traverse backward from the goal to the start
                path.append(node.action)
                node = node.parent
            return path[::-1], explored_states  # Reverse the path to get the correct order

        # Skip already explored states
        if node.state in explored:
            continue

        # Mark the current node as explored
        explored.add(node.state)
        explored_states.append(node.state)

        # Add valid neighbors to the frontier
        for action, state in maze.neighbors(node.state):
            if state not in explored:
                # Calculate the priority based on the algorithm
                if algorithm == "greedy":
                    # Greedy uses the heuristic: Manhattan distance to the goal
                    priority = abs(state[0] - maze.goal[0]) + abs(state[1] - maze.goal[1])
                elif algorithm == "astar":
                    # A* combines the cost so far and the heuristic
                    priority = node.cost + 1 + abs(state[0] - maze.goal[0]) + abs(state[1] - maze.goal[1])
                # Add the neighbor to the frontier
                heapq.heappush(frontier, (priority, Node(state, node, action, node.cost + 1)))

    # If no path is found, raise an exception
    raise Exception("No path exists")

if __name__ == "__main__":
    # **Initialize the Maze**
    maze = Maze("maze.txt")

    # **Greedy Best-First Search**
    try:
        path, explored = solve(maze, "greedy")
        print("Greedy Path:", path)
        visualize(maze, path, explored, "greedy_solution.png")  # Save visualization
    except Exception as e:
        print(e)

    # **A* Search**
    try:
        path, explored = solve(maze, "astar")
        print("A* Path:", path)
        visualize(maze, path, explored, "astar_solution.png")  # Save visualization
    except Exception as e:
        print(e)