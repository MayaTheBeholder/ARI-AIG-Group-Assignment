# Importing required libraries
import matplotlib.pyplot as plt  # For plotting the visualization
import numpy as np  # For handling and manipulating arrays
from matplotlib.colors import ListedColormap  # To define custom colors for the maze visualization

# ### Visualize Function
def visualize(maze, path, explored, filename):
    """
    **Visualizes the maze, the path, and the explored states.**
    - Highlights:
        - Walls: Black
        - Start: Green
        - Goal: Red
        - Path: Blue
        - Explored States: Yellow
    - Saves the visualization as an image file.
    """
    # Convert the maze grid into a NumPy array for easier manipulation
    grid = np.array(maze.grid)
    display = np.zeros_like(grid)  # Create a blank grid for visualization

    # #### Step 1: Mark Walls
    # Iterate over all wall positions and mark them on the display grid
    for wall in maze.walls:
        display[wall[0], wall[1]] = 1  # `1` represents a wall

    # #### Step 2: Mark Start and Goal Positions
    # Mark the start position
    display[maze.start[0], maze.start[1]] = 2  # `2` represents the start
    # Mark the goal position
    display[maze.goal[0], maze.goal[1]] = 3  # `3` represents the goal

    # #### Step 3: Mark the Path (if a path exists)
    if path:
        current = maze.start  # Start from the initial position
        for action in path:  # Traverse the actions in the path
            if action == "up":
                current = (current[0]-1, current[1])  # Move up
            elif action == "down":
                current = (current[0]+1, current[1])  # Move down
            elif action == "left":
                current = (current[0], current[1]-1)  # Move left
            elif action == "right":
                current = (current[0], current[1]+1)  # Move right
            # Mark the path position
            display[current[0], current[1]] = 4  # `4` represents the path

    # #### Step 4: Mark Explored States
    for state in explored:  # Iterate over all explored states
        if display[state[0], state[1]] == 0:  # Ensure the state is not already marked
            display[state[0], state[1]] = 5  # `5` represents explored positions

    # #### Step 5: Define Custom Colors for Visualization
    cmap = ListedColormap([
        'white',  # `0`: Open space
        'black',  # `1`: Walls
        'green',  # `2`: Start position
        'red',    # `3`: Goal position
        'blue',   # `4`: Path
        'yellow'  # `5`: Explored positions
    ])

    # #### Step 6: Plot the Maze
    plt.figure(figsize=(8, 8))  # Set the figure size
    plt.imshow(display, cmap=cmap)  # Render the display grid with the custom colormap
    plt.xticks([]), plt.yticks([])  # Remove axis ticks for a cleaner look

    # #### Step 7: Save the Visualization
    plt.savefig(filename)  # Save the plot as an image file
    plt.close()  # Close the figure to free up memory