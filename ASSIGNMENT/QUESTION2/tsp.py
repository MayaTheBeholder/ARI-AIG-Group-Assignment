# Importing necessary libraries
import random
import math
from visualisation import plot_route, plot_convergence  # Custom visualization functions

class TSP:
    """
    Represents the Traveling Salesman Problem (TSP).
    Stores the distance matrix and provides route distance calculation.
    """
    def __init__(self, distances):
        self.distances = distances
        self.size = len(distances)

    def route_distance(self, route):
        """Calculates the total distance of a given route."""
        return sum(self.distances[route[i]][route[i+1]] for i in range(len(route)-1))

class SimulatedAnnealing:
    """
    Uses the Simulated Annealing algorithm to solve the TSP.
    """
    def __init__(self, tsp, temp=10000, cooling=0.003):
        self.tsp = tsp
        self.temp = temp
        self.cooling = cooling
        self.history = []

    def initial_route(self):
        """Generates an initial random route."""
        route = list(range(self.tsp.size))
        random.shuffle(route)
        return route + [route[0]]

    def neighbor_route(self, route):
        """Generates a neighboring route by swapping two cities."""
        i, j = random.sample(range(1, len(route)-1), 2)
        new_route = route.copy()
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def acceptance_prob(self, old_cost, new_cost):
        """Calculates the probability of accepting a new solution."""
        if new_cost > old_cost:
            return math.exp((old_cost - new_cost) / self.temp)
        return 1.0

    def solve(self):
        """Solves the TSP using Simulated Annealing."""
        current = self.initial_route()
        best = current.copy()

        while self.temp > 1:
            neighbor = self.neighbor_route(current)

            if self.acceptance_prob(self.tsp.route_distance(current), 
                                  self.tsp.route_distance(neighbor)) > random.random():
                current = neighbor
                if self.tsp.route_distance(current) < self.tsp.route_distance(best):
                    best = current.copy()

            self.history.append(self.tsp.route_distance(current))
            self.temp *= 1 - self.cooling

        return best

if __name__ == "__main__":
    # Distance matrix for 10 cities
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

    tsp = TSP(distances)
    sa = SimulatedAnnealing(tsp)

    initial = sa.initial_route()
    print("Initial route distance:", tsp.route_distance(initial))

    solution = sa.solve()
    print("Optimized route distance:", tsp.route_distance(solution))

    # Visualization
    plot_convergence(sa.history)
    plot_route(solution, tsp.route_distance(solution))