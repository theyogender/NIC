import random
import numpy as np

def generate_random_tour(num_cities):
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

def fitness(tour, distance_matrix):
    return sum(distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1)) + distance_matrix[tour[-1]][tour[0]]

def partially_mapped_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end+1] = parent1[start:end+1]
    mapping = {parent1[i]: parent2[i] for i in range(start, end+1)}
    
    for i in range(size):
        if child[i] == -1:
            val = parent2[i]
            while val in mapping:
                val = mapping[val]
            child[i] = val
    
    return child

def edge_recombination_crossover(parent1, parent2):
    size = len(parent1)
    edge_table = {i: set() for i in range(size)}
    
    for p in [parent1, parent2]:
        for i in range(size):
            edge_table[p[i]].update({p[(i-1) % size], p[(i+1) % size]})
    
    current = random.choice(parent1)
    child = [current]
    
    while len(child) < size:
        for key in edge_table:
            edge_table[key].discard(current)
        
        if edge_table[current]:
            next_city = min(edge_table[current], key=lambda x: len(edge_table[x]))
        else:
            next_city = random.choice([c for c in range(size) if c not in child])
        
        child.append(next_city)
        current = next_city
    
    return child

def mutate(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i], tour[j] = tour[j], tour[i]
    return tour

def genetic_algorithm_tsp(distance_matrix, population_size=100, generations=500, crossover_type="PMX"):
    num_cities = len(distance_matrix)
    population = [generate_random_tour(num_cities) for _ in range(population_size)]
    
    for generation in range(generations):
        population = sorted(population, key=lambda tour: fitness(tour, distance_matrix))
        best_tour = population[0]
        print(f"Generation {generation+1}: Best tour: {[cities[i] for i in best_tour]}, Fitness: {fitness(best_tour, distance_matrix)}")
        new_population = population[:10]
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:50], 2)
            if crossover_type == "PMX":
                child = partially_mapped_crossover(parent1, parent2)
            else:
                child = edge_recombination_crossover(parent1, parent2)
            if random.random() < 0.2:
                child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    return min(population, key=lambda tour: fitness(tour, distance_matrix))

# User input for number of cities
num_cities = int(input("Enter the number of cities: "))
cities = []
print("Enter city names:")
for _ in range(num_cities):
    cities.append(input().strip())

# Initialize distance matrix
distance_matrix = np.full((num_cities, num_cities), np.inf)

print("Enter the distance between cities (e.g., A B 10). If no connection, press Enter.")
while True:
    entry = input("Enter connection (or press Enter to finish): ").strip()
    if not entry:
        break
    city1, city2, distance = entry.split()
    i, j = cities.index(city1), cities.index(city2)
    distance = int(distance)
    distance_matrix[i][j] = distance_matrix[j][i] = distance

# Replace np.inf with a large number for unconnected paths
distance_matrix[distance_matrix == np.inf] = 99999

# User selects crossover method
crossover_choice = input("Select crossover method (PMX for Partially Mapped, ERX for Edge Recombination): ").strip().upper()
if crossover_choice not in ["PMX", "ERX"]:
    print("Invalid choice! Defaulting to PMX.")
    crossover_choice = "PMX"

# Run Genetic Algorithm
best_tour = genetic_algorithm_tsp(distance_matrix, crossover_type=crossover_choice)
best_tour_cities = [cities[i] for i in best_tour]
print(f"Final Best tour using {crossover_choice}:", best_tour_cities, "Fitness:", fitness(best_tour, distance_matrix))


