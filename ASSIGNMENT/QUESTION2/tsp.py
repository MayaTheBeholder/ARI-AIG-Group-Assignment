# Importing necessary libraries
import random  # To generate random values
import math  # For mathematical computations
from visualization import plot_route, plot_convergence  # Custom visualization functions

# ### Traveling Salesman Problem (TSP) Class
class TSP:
    """
    **Represents the Traveling Salesman Problem (TSP).**
    - Stores the distance matrix and the number of cities.
    - Provides functionality to calculate the total distance of a given route.
    """
    def __init__(self, distances):
        """
        Initializes the TSP with the given distance matrix.
        - `distances`: 2D list where distances[i][j] represents the distance between city `i` and city `j`.
        """
        self.distances = distances
        self.size = len(distances)  # Number of cities

    def route_distance(self, route):
        """
        **Calculates the total distance of a given route.**
        - A route is a list of city indices.
        """
        return sum(self.distances[route[i]][route[i+1]] for i in range(len(route)-1))

# ### Simulated Annealing Solver for TSP
class SimulatedAnnealing:
    """
    **Uses the Simulated Annealing algorithm to solve the TSP.**
    - Attempts to find an optimized route through all cities.
    """
    def __init__(self, tsp, temp=10000, cooling=0.003):
        """
        Initializes the solver with the given TSP instance and algorithm parameters.
        - `tsp`: An instance of the TSP class.
        - `temp`: Initial temperature for the algorithm (higher values allow more exploration).
        - `cooling`: Cooling rate to reduce the temperature gradually.
        """
        self.tsp = tsp
        self.temp = temp  # Initial temperature
        self.cooling = cooling  # Cooling rate
        self.history = []  # Tracks the cost of solutions over iterations

    def initial_route(self):
        """
        **Generates an initial random route.**
        - Ensures the route starts and ends at the same city.
        """
        route = list(range(self.tsp.size))  # List of city indices
        random.shuffle(route)  # Randomly shuffle the order of cities
        return route + [route[0]]  # Make it a round trip

    def neighbor_route(self, route):
        """
        **Generates a neighboring route by swapping two cities.**
        - Swaps two random cities in the route (excluding the start/end city).
        """
        i, j = random.sample(range(1, len(route)-1), 2)  # Random indices
        new_route = route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]  # Swap cities
        return new_route

    def acceptance_prob(self, old_cost, new_cost):
        """
        **Calculates the probability of accepting a new solution.**
        - Accepts worse solutions probabilistically based on the temperature.
        """
        if new_cost > old_cost:  # New solution is better
            return math.exp((old_cost - new_cost) / self.temp)  # Acceptance probability
        return 1.0  # Always accept better solutions

    def solve(self):
        """
        **Solves the TSP using the Simulated Annealing algorithm.**
        - Starts with a random route and iteratively improves it.
        - Returns the best route found.
        """
        current = self.initial_route()  # Start with an initial random route
        best = current.copy()  # Track the best route found

        while self.temp > 1:  # Continue until the temperature is very low
            neighbor = self.neighbor_route(current)  # Generate a neighboring solution

            # Decide whether to accept the neighboring solution
            if self.acceptance_prob(self.tsp.route_distance(current), 
                                    self.tsp.route_distance(neighbor)) > random.random():
                current = neighbor  # Accept the neighbor as the current solution

                # Update the best solution if the current solution is better
                if self.tsp.route_distance(current) < self.tsp.route_distance(best):
                    best = current.copy()

            # Record the cost of the current solution
            self.history.append(self.tsp.route_distance(current))
            # Decrease the temperature
            self.temp *= 1 - self.cooling

        return best  # Return the best solution found

# ### Main Functionality
if __name__ == "__main__":
    """
    **Solves a sample TSP problem using Simulated Annealing.**
    - The distance matrix represents distances between cities.
    """
    # Distance matrix for 10 cities (example data)
    # Indices: 0 = Windhoek, 1 = Swakopmund, etc.
    distances = [
        [0, 361, 395, 249, 433, 459, 268, 497, 678, 712],
        [361, 0, 35.5, 379, 562, 589, 541, 859, 808, 779],
        [395, 35.5, 0, 413, 597, 623, 511, 732, 884, 855],
        [249, 379, 413, 0, 260, 183, 519, 768, 514, 485],
        [433, 562, 597, 260, 0, 60, 682, 921, 254, 288],
        [459, 589, 623, 183, 60, 0, 708, 947, 308, 342],
        [268, 541, 511, 519, 682, 708, 0, 231, 909, 981],
        [497, 859, 732, 768, 921, 947, 231, 0, 1175, 1210],
        [678, 808, 884, 514, 254, 308, 909, 1175, 0, 30],
        [712, 779, 855, 485, 288, 342, 981, 1210, 30, 0]
    ]

    # Initialize the TSP instance
    tsp = TSP(distances)

    # Initialize the Simulated Annealing solver
    sa = SimulatedAnnealing(tsp)

    # Generate an initial random route
    initial = sa.initial_route()
    print("Initial route distance:", tsp.route_distance(initial))

    # Solve the TSP using Simulated Annealing
    solution = sa.solve()
    print("Optimized route distance:", tsp.route_distance(solution))

    # Plot the convergence history and the final route
    plot_convergence(sa.history)  # Plot how the solution improved over iterations
    plot_route(solution, tsp.route_distance(solution))  # Visualize the final route