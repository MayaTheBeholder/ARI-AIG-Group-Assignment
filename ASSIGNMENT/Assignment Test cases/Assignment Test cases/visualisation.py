import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def visualize(maze, path, explored, filename):
    grid = np.array(maze.grid)
    display = np.zeros_like(grid)
    
    # walls
    for wall in maze.walls:
        display[wall[0], wall[1]] = 1
    
    #start & goal
    display[maze.start[0], maze.start[1]] = 2
    display[maze.goal[0], maze.goal[1]] = 3
    
    # path
    if path:
        current = maze.start
        for action in path:
            if action == "up":
                current = (current[0]-1, current[1])
            elif action == "down":
                current = (current[0]+1, current[1])
            elif action == "left":
                current = (current[0], current[1]-1)
            elif action == "right":
                current = (current[0], current[1]+1)
            display[current[0], current[1]] = 4
    
    # explored
    for state in explored:
        if display[state[0], state[1]] == 0:
            display[state[0], state[1]] = 5
    
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'blue', 'yellow'])
    plt.figure(figsize=(8,8))
    plt.imshow(display, cmap=cmap)
    plt.xticks([]), plt.yticks([])
    plt.savefig(filename)
    plt.close()