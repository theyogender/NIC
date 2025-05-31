import numpy as np
import random

def fitness(route):
    n = len(route)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(route[i] - route[j]) == abs(i - j):  
                conflicts += 1
    return conflicts

class ACO_NQueens:
    def __init__(self, n, num_ants, evaporation_rate, alpha, beta, q, iterations):
        self.n = n
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.iterations = iterations

        self.pheromone = np.ones((n, n)) 

    def run(self):
        best_route = None
        best_fitness = float('inf')

        for iteration in range(self.iterations):
            all_routes = self.generate_routes()
            self.update_pheromone(all_routes)

            for route, fit in all_routes:
                if fit < best_fitness:
                    best_fitness = fit
                    best_route = route

                if best_fitness == 0: 
                    return best_route, best_fitness

        return best_route, best_fitness

    def generate_routes(self):
        all_routes = []
        for _ in range(self.num_ants):
            route = self.construct_route()
            fit = fitness(route)
            all_routes.append((route, fit))
        return all_routes

    def construct_route(self):
        route = []
        visited_columns = set()

        for row in range(self.n):
            col = self.select_next_column(row, visited_columns)
            route.append(col)
            visited_columns.add(col)

        return route

    def select_next_column(self, row, visited_columns):
        columns = [c for c in range(self.n) if c not in visited_columns]
        probabilities = []
        total = 0.0

        for col in columns:
            pheromone = self.pheromone[row][col] ** self.alpha
            heuristic = 1.0  
            value = pheromone * (heuristic ** self.beta)
            probabilities.append(value)
            total += value

        probabilities = [p / total for p in probabilities] if total > 0 else [1 / len(columns)] * len(columns)
        return random.choices(columns, weights=probabilities, k=1)[0]

    def update_pheromone(self, all_routes):
        self.pheromone *= (1 - self.evaporation_rate)

        for route, fit in all_routes:
            if fit == 0:
                pheromone_to_add = self.q
            else:
                pheromone_to_add = self.q / fit

            for row in range(self.n):
                col = route[row]
                self.pheromone[row][col] += pheromone_to_add

n = int(input("Enter the value of N (for N-Queens): "))
num_ants = int(input("Enter number of ants: "))
evaporation_rate = float(input("Enter evaporation rate (0-1): "))
alpha = float(input("Enter alpha (pheromone importance): "))
beta = float(input("Enter beta (heuristic importance): "))
q = float(input("Enter Q (pheromone deposit factor): "))
iterations = int(input("Enter number of iterations: "))

aco = ACO_NQueens(n, num_ants, evaporation_rate, alpha, beta, q, iterations)
best_solution, min_conflicts = aco.run()

print("\nBest Queen Positions (Matrix Format):")
for row in range(n):
    line = ['Q' if best_solution[row] == col else '.' for col in range(n)]
    print(' '.join(line))
print("\nNumber of Conflicts:", min_conflicts)

'''OUTPUT
Enter the value of N (for N-Queens): 8
Enter number of ants: 8
Enter evaporation rate (0-1): 0.5
Enter alpha (pheromone importance): 1
Enter beta (heuristic importance): 1
Enter Q (pheromone deposit factor): 1
Enter number of iterations: 100

Best Queen Positions (Matrix Format):
. . . . Q . . .
. . . . . . . Q
. . . Q . . . .
Q . . . . . . .
. . . . . . Q .
. Q . . . . . .
. . . . . Q . .
. . Q . . . . .

Number of Conflicts: 0
'''