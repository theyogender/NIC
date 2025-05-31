import numpy as np
import random
import pprint

class AntColony:
    def __init__(self, dist_matrix, n_ants, n_iterations, decay=0.5, alpha=1, beta=2):
        self.dist_matrix = dist_matrix
        self.pheromone = np.ones(self.dist_matrix.shape) / len(dist_matrix)
        self.n_cities = len(dist_matrix)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha  # influence of pheromone
        self.beta = beta    # influence of distance

    def run(self):
        best_path = None
        best_dist = float("inf")

        for it in range(self.n_iterations):
            print(f"\n====== Iteration {it+1} ======")
            all_paths = self.construct_all_paths()

            print("\nAnt Paths and Distances:")
            for i, (path, dist) in enumerate(all_paths):
                print(f"Ant {i+1}: Path {path} -> Distance = {dist}")

            self.spread_pheromone(all_paths)
            self.pheromone *= self.decay

            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < best_dist:
                best_path = shortest_path[0]
                best_dist = shortest_path[1]

            print("\nUpdated Pheromone Matrix:")
            print(np.round(self.pheromone, 4))

        return best_path, best_dist

    def spread_pheromone(self, all_paths):
        print("\nPheromone Spread:")
        for path, dist in all_paths:
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                delta = 1.0 / dist
                self.pheromone[a][b] += delta
                self.pheromone[b][a] += delta
                print(f"Pheromone added on edge ({a},{b}) and ({b},{a}): +{delta:.4f}")

    def construct_all_paths(self):
        all_paths = []
        for ant in range(self.n_ants):
            path = []
            visited = set()
            current_city = random.randint(0, self.n_cities - 1)
            path.append(current_city)
            visited.add(current_city)

            while len(path) < self.n_cities:
                next_city = self.pick_next_city(current_city, visited)
                path.append(next_city)
                visited.add(next_city)
                current_city = next_city

            path.append(path[0])  # return to start
            total_dist = self.calculate_total_distance(path)
            all_paths.append((path, total_dist))

        return all_paths

    def pick_next_city(self, current_city, visited):
        pheromone = self.pheromone[current_city]
        distances = self.dist_matrix[current_city]

        probabilities = []
        for city in range(self.n_cities):
            if city not in visited:
                tau = pheromone[city] ** self.alpha
                eta = (1.0 / distances[city]) ** self.beta
                prob = tau * eta
                probabilities.append((city, prob))
            else:
                probabilities.append((city, 0))

        total = sum(prob for _, prob in probabilities)
        probs = [(city, prob / total if total > 0 else 0) for city, prob in probabilities]

        print(f"\nCurrent City: {current_city}")
        print("Probabilities for next move:")
        for city, prob in probs:
            if city not in visited:
                print(f"To City {city} -> Prob = {prob:.4f}")

        cities = [city for city, prob in probs]
        weights = [prob for city, prob in probs]
        next_city = random.choices(cities, weights=weights)[0]
        return next_city

    def calculate_total_distance(self, path):
        total = 0
        for i in range(len(path) - 1):
            total += self.dist_matrix[path[i]][path[i+1]]
        return total


# === INPUT SECTION ===
n = int(input("Enter the number of cities: "))
dist_matrix = np.full((n, n), np.inf)

print("\nEnter the path details in format: city1 city2 distance")
print("Type 'done' when finished.")
while True:
    entry = input("Path (or 'done'): ")
    if entry.strip().lower() == "done":
        break
    try:
        c1, c2, d = map(int, entry.strip().split())
        dist_matrix[c1][c2] = d
        dist_matrix[c2][c1] = d  # Assuming undirected path
    except:
        print("Invalid input, please enter again.")

# Replace inf with large number for compatibility
dist_matrix[dist_matrix == np.inf] = 1e9

# === RUN ACO ===
aco = AntColony(dist_matrix, n_ants=5, n_iterations=5, decay=0.5, alpha=1, beta=2)
best_path, best_dist = aco.run()

print("\n==============================")
print("Best Path Found:")
print(best_path)
print(f"Total Distance: {best_dist}")
