import numpy as np
import random

def calculate_distance(tour, distance_matrix):
    return sum(distance_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + distance_matrix[tour[-1], tour[0]]

def cyclic_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    index = 0
    start = parent1[0]
    while True:
        child[index] = parent1[index]
        index = parent1.index(parent2[index])
        if parent1[index] == start:
            break
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

def mutate(tour, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def tournament_selection(population, fitness, k=3):
    selected = random.sample(list(zip(population, fitness)), k)
    return min(selected, key=lambda x: x[1])[0]

def genetic_algorithm_tsp(distance_matrix, population_size=50, generations=1000, mutation_rate=0.1):
    num_cities = len(distance_matrix)
    population = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]
    
    for _ in range(generations):
        fitness = [calculate_distance(tour, distance_matrix) for tour in population]
        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            child1 = cyclic_crossover(parent1, parent2)
            child2 = cyclic_crossover(parent2, parent1)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        population = new_population
    
    best_tour = min(population, key=lambda tour: calculate_distance(tour, distance_matrix))
    return best_tour, calculate_distance(best_tour, distance_matrix)

import numpy as np
import random

def calculate_distance(tour, distance_matrix):
    return sum(distance_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + distance_matrix[tour[-1], tour[0]]

def cyclic_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    index = 0
    start = parent1[0]
    while True:
        child[index] = parent1[index]
        index = parent1.index(parent2[index])
        if parent1[index] == start:
            break
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    return child

def mutate(tour, mutation_rate=0.1):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def tournament_selection(population, fitness, k=3):
    selected = random.sample(list(zip(population, fitness)), k)
    return min(selected, key=lambda x: x[1])[0]

def genetic_algorithm_tsp(distance_matrix, population_size=50, generations=1000, mutation_rate=0.1):
    num_cities = len(distance_matrix)
    population = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]
    
    for gen in range(generations):
        fitness = [calculate_distance(tour, distance_matrix) for tour in population]
        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            child1 = cyclic_crossover(parent1, parent2)
            child2 = cyclic_crossover(parent2, parent1)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
        population = new_population
        
        best_tour = min(population, key=lambda tour: calculate_distance(tour, distance_matrix))
        best_distance = calculate_distance(best_tour, distance_matrix)
        print(f"Generation {gen+1}: Best Distance = {best_distance}, Best Tour = {best_tour}")
    
    return best_tour, best_distance

# User Input for Distance Matrix
num_cities = int(input("Enter the number of cities: "))
distance_matrix = np.zeros((num_cities, num_cities))

print("Enter the distances between cities:")
for i in range(num_cities):
    for j in range(i + 1, num_cities):
        distance = int(input(f"Enter the distance between city {i+1} and city {j+1}: "))
        distance_matrix[i][j] = distance
        distance_matrix[j][i] = distance  # Since the distance matrix is symmetric

print("Initial Distance Matrix:")
print(distance_matrix)

best_tour, best_distance = genetic_algorithm_tsp(distance_matrix)
print("\nFinal Best Tour:", best_tour)
print("Final Best Distance:", best_distance)


