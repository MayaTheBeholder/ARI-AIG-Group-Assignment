import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(history):
    plt.figure(figsize=(10,5))
    plt.plot(history)
    plt.title("Simulated Annealing Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Route Distance")
    plt.savefig("convergence.png")
    plt.close()

def plot_route(route, distance):
    angles = np.linspace(0, 2*np.pi, len(route), endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    plt.figure(figsize=(8,8))
    plt.plot(x, y, 'o-')
    for i, town in enumerate(route[:-1]):
        plt.text(x[i], y[i]+0.05, str(town), ha='center')
    
    plt.title(f"TSP Route - Distance: {distance:.2f} km")
    plt.axis('equal')
    plt.savefig("route.png")
    plt.close()