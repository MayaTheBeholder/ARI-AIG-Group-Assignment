# Importing necessary libraries
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations

# ### Plotting the Convergence of Simulated Annealing
def plot_convergence(history):
    """
    **Plots the convergence of the Simulated Annealing algorithm.**
    - Visualizes how the solution improves over iterations.
    
    **Parameters:**
    - `history`: A list of route distances recorded during the algorithm's execution.
    """
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(history)  # Plot the route distances over iterations
    plt.title("Simulated Annealing Convergence")  # Add a title
    plt.xlabel("Iteration")  # Label for the x-axis
    plt.ylabel("Route Distance")  # Label for the y-axis
    plt.savefig("convergence.png")  # Save the plot as an image
    plt.close()  # Close the plot to prevent overlapping with future plots

# ### Plotting the Optimized Traveling Salesman Problem (TSP) Route
def plot_route(route, distance):
    """
    **Visualizes the optimized route for the Traveling Salesman Problem (TSP).**
    - Depicts the cities as points on a circle and connects them in the order of the route.
    
    **Parameters:**
    - `route`: A list of city indices representing the order of travel.
    - `distance`: The total distance of the route.
    """
    # Calculate the angles to evenly space the cities on a circle
    angles = np.linspace(0, 2 * np.pi, len(route), endpoint=False)
    x = np.cos(angles)  # x-coordinates for cities
    y = np.sin(angles)  # y-coordinates for cities

    # Create the plot
    plt.figure(figsize=(8, 8))  # Set the figure size
    plt.plot(x, y, 'o-', markersize=8)  # Plot the route as a connected line
    for i, town in enumerate(route[:-1]):  # Label each city (excluding the last duplicate city)
        plt.text(x[i], y[i] + 0.05, str(town), ha='center', fontsize=10)  # Add city labels

    # Add a title with the total distance
    plt.title(f"TSP Route - Distance: {distance:.2f} km")
    plt.axis('equal')  # Ensure equal scaling for the x and y axes
    plt.savefig("route.png")  # Save the plot as an image
    plt.close()  # Close the plot to prevent overlapping with future plots