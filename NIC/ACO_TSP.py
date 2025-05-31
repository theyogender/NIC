import numpy as np
import random

def fitness(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

class ACO_TSP:
    def __init__(self, distance_matrix, num_ants, evaporation_rate, alpha, beta, q, iterations):
        self.distance_matrix = distance_matrix
        self.num_nodes = len(distance_matrix)
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.iterations = iterations

        self.pheromone = np.ones((self.num_nodes, self.num_nodes)) / self.num_nodes
        self.heuristic = 1 / (np.array(distance_matrix) + 1e-10)

    def run(self):
        best_route = None
        best_distance = float('inf')

        for _ in range(self.iterations):
            all_routes = self.generate_routes()
            self.update_pheromone(all_routes)

            for route, distance in all_routes:
                if distance < best_distance:
                    best_distance = distance
                    best_route = route

        return best_route, best_distance

    def generate_routes(self):
        all_routes = []
        for _ in range(self.num_ants):
            route = self.construct_route()
            distance = fitness(route, self.distance_matrix)
            all_routes.append((route, distance))
        return all_routes

    def construct_route(self):
        visited = set()
        current_node = random.randint(0, self.num_nodes - 1)
        start_node = current_node
        route = [current_node]
        visited.add(current_node)

        while len(visited) < self.num_nodes:
            next_node = self.select_next_node(current_node, visited)
            route.append(next_node)
            visited.add(next_node)
            current_node = next_node

        route.append(start_node)
        return route

    def select_next_node(self, current_node, visited):
        unvisited = [node for node in range(self.num_nodes) if node not in visited]

        probabilities = []
        total = 0.0

        for node in unvisited:
            pheromone = self.pheromone[current_node][node] ** self.alpha
            heuristic = self.heuristic[current_node][node] ** self.beta
            probabilities.append(pheromone * heuristic)
            total += pheromone * heuristic

        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1 / len(unvisited)] * len(unvisited)

        return random.choices(unvisited, weights=probabilities, k=1)[0]

    def update_pheromone(self, all_routes):
        self.pheromone *= (1 - self.evaporation_rate)

        for route, distance in all_routes:
            pheromone_to_add = self.q / distance
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                self.pheromone[from_node][to_node] += pheromone_to_add
                self.pheromone[to_node][from_node] += pheromone_to_add 

num_nodes = int(input("Enter number of nodes: "))
distance_matrix = []
print("Enter the distance matrix:")
for i in range(num_nodes):
    row = list(map(int, input().split()))
    for j in range(num_nodes):
        if i != j and row[j] == 0:
            row[j] = float('inf')
    distance_matrix.append(row)

num_ants = int(input("Enter number of ants: "))
evaporation_rate = float(input("Enter evaporation rate (0-1): "))
alpha = float(input("Enter alpha (pheromone importance): "))
beta = float(input("Enter beta (heuristic importance): "))
q = float(input("Enter Q (pheromone deposit factor): "))
iterations = int(input("Enter number of iterations: "))

aco = ACO_TSP(distance_matrix, num_ants, evaporation_rate, alpha, beta, q, iterations)
best_route, min_distance = aco.run()

print("\nBest Route:", " -> ".join(map(str, best_route)))
print("Total Distance:", min_distance)


'''OUTPUT
Enter number of nodes: 5
Enter the distance matrix:
0 4 3 0 2
4 0 2 3 0
3 2 0 1 2
0 3 1 0 4
2 0 2 4 0
Enter number of ants: 3
Enter evaporation rate (0-1): 0.6
Enter alpha (pheromone importance): 1
Enter beta (heuristic importance): 1
Enter Q (pheromone deposit factor): 1
Enter number of iterations: 100

Best Route: 3 -> 2 -> 4 -> 0 -> 1 -> 3
Total Distance: 12
'''