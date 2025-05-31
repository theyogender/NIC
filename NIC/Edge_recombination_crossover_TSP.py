import random
import numpy as np

def total_distance(route, distance_matrix):
    distance = 0
    for i in range(len(route)):
        city1 = route[i]
        city2 = route[(i + 1) % len(route)]
        distance += distance_matrix[city1][city2]
    return distance

def generate_population(num_cities, population_size):
    return [random.sample(range(num_cities), num_cities) for _ in range(population_size)]

def fitness(route, distance_matrix):
    return 1 / total_distance(route, distance_matrix)

def edge_recombination(parent1, parent2):
    num_cities = len(parent1)

    adj_list = {city: set() for city in parent1}

    for i in range(num_cities):
        adj_list[parent1[i]].update([
            parent1[(i - 1) % num_cities], parent1[(i + 1) % num_cities]
        ])
        adj_list[parent2[i]].update([
            parent2[(i - 1) % num_cities], parent2[(i + 1) % num_cities]
        ])

    current_city = random.choice(parent1) 
    child = [current_city]

    while len(child) < num_cities:
        neighbors = adj_list[current_city]

        for city in adj_list:
            adj_list[city].discard(current_city)

        if neighbors:
            next_city = min(neighbors, key=lambda x: len(adj_list[x]))
        else:
            next_city = random.choice([c for c in range(num_cities) if c not in child])

        child.append(next_city)
        current_city = next_city

    return child

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
            
            child = edge_recombination(parent1, parent2)
            
            mutated_child = mutate(child, mutation_rate)
            
            new_population.append(mutated_child)

        population = new_population

        best_route = min(population, key=lambda x: total_distance(x, distance_matrix))
        best_distance = total_distance(best_route, distance_matrix)

    best_route = min(population, key=lambda x: total_distance(x, distance_matrix))
    best_distance = total_distance(best_route, distance_matrix)

    # Ensure route is printed as a cycle
    print("\nFinal Best Route:", " -> ".join(map(str, best_route + [best_route[0]])))
    print("Total Distance:", best_distance)


num_cities = int(input("Enter the number of cities: "))

print("Enter the distance matrix:")
distance_matrix = []
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


''' OUTPUT
Enter the number of cities: 5
Enter the distance matrix:
0 4 3 0 2
4 0 2 3 0
3 2 0 1 2
0 3 1 0 4
2 0 2 4 0

Enter population size: 100
Enter mutation rate (0-1): 0.2
Enter maximum number of generations: 100

Final Best Route: 4 -> 0 -> 1 -> 3 -> 2 -> 4
Total Distance: 12
'''