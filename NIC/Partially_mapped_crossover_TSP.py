import random
import numpy as np

def total_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i]][route[i + 1]]
    # Add return trip to starting city
    distance += distance_matrix[route[-1]][route[0]]
    return distance

def generate_population(num_cities, population_size):
    return [random.sample(range(num_cities), num_cities) for _ in range(population_size)]

def fitness(route, distance_matrix):
    return 1 / total_distance(route, distance_matrix)

def pmx_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child1, child2 = [-1]*size, [-1]*size
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    def fill_child(child, parentA, parentB):
        for i in range(len(child)):
            if child[i] == -1:
                gene = parentB[i]
                while gene in child:
                    gene = parentB[parentA.index(gene)]
                child[i] = gene

    fill_child(child1, parent1, parent2)
    fill_child(child2, parent2, parent1)

    return child1, child2

def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def rank_selection(population, distance_matrix):
    population_size = len(population)
    sorted_population = sorted(population, key=lambda x: fitness(x, distance_matrix), reverse=True)

    ranks = list(range(1, population_size + 1))
    total_rank = sum(ranks)
    probabilities = [rank / total_rank for rank in ranks]

    selected = random.choices(sorted_population, weights=probabilities, k=2)
    return selected[0], selected[1]

def genetic_algorithm(num_cities, distance_matrix, population_size, mutation_rate, generations):
    population = generate_population(num_cities, population_size)

    for generation in range(generations):
        new_population = []

        while len(new_population) < population_size:
            parent1, parent2 = rank_selection(population, distance_matrix)
            child1, child2 = pmx_crossover(parent1, parent2)
            
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))

        population = new_population

    best_route = min(population, key=lambda x: total_distance(x, distance_matrix))
    best_distance = total_distance(best_route, distance_matrix)

    print("\nFinal Best Route:", " -> ".join(map(str, best_route + [best_route[0]])))
    print("Total Distance:", best_distance)

# --- Input Section ---
num_cities = int(input("Enter the number of cities: "))

distance_matrix = []
print("Enter the distance matrix:")
for i in range(num_cities):
    row = list(map(int, input().split()))
    for j in range(num_cities):
        if i != j and row[j] == 0:
            row[j] = float('inf')
    distance_matrix.append(row)

population_size = int(input("\nEnter population size: "))
mutation_rate = float(input("Enter mutation rate (0-1): "))
generations = int(input("Enter maximum number of generations: "))

genetic_algorithm(num_cities, distance_matrix, population_size, mutation_rate, generations)


'''OUTPUT
Enter the number of cities: 5
Enter the distance matrix:
0 1 2 3 4
1 0 4 3 2
2 4 0 2 2
3 3 2 0 0
4 2 2 0 0

Enter population size: 100
Enter mutation rate (0-1): 0.2
Enter maximum number of generations: 100

Final Best Route: 0 -> 1 -> 4 -> 2 -> 3 -> 0
Total Distance: 10
'''