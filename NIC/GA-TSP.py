import numpy as np
import random
from itertools import permutations

def fitness(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) + distance_matrix[route[-1]][route[0]]

def initialize_population(pop_size, num_nodes):
    population = [random.sample(range(1, num_nodes), num_nodes - 1) for _ in range(pop_size)]
    return population

def cycle_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [-1] * size, [-1] * size
    
    def find_cycle(start, p1, p2):
        cycle = []
        val = p1[start]
        while val not in cycle:
            cycle.append(val)
            val = p1[p2.index(val)]
        return cycle
    
    cycle = find_cycle(0, parent1, parent2)
    for i in cycle:
        child1[parent1.index(i)] = i
        child2[parent2.index(i)] = i
    
    for i in range(size):
        if child1[i] == -1:
            child1[i] = parent2[i]
        if child2[i] == -1:
            child2[i] = parent1[i]
    
    return child1, child2

def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]

def genetic_algorithm(num_nodes, distance_matrix, pop_size, mutation_rate, max_generations):
    population = initialize_population(pop_size, num_nodes)
    best_route = None
    best_fitness = float('inf')
    
    for generation in range(max_generations):
        population.sort(key=lambda x: fitness([0] + x + [0], distance_matrix))
        new_population = population[:2]
        
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(population[:pop_size // 2], 2)
            child1, child2 = cycle_crossover(parent1, parent2)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population[:pop_size]
        current_best = min(population, key=lambda x: fitness([0] + x + [0], distance_matrix))
        current_fitness = fitness([0] + current_best + [0], distance_matrix)
        
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_route = [0] + current_best + [0]
        
        if best_route == population[0]:
            break
    
    return best_route, best_fitness

num_nodes = int(input("Enter number of nodes: "))
distance_matrix = []
print("Enter the distance matrix:")
for i in range(num_nodes):
    row = list(map(int, input().split()))
    for j in range(num_nodes):
        if i != j and row[j] == 0:
            row[j] = float('inf')
    distance_matrix.append(row)

pop_size = int(input("Enter population size: "))
mutation_rate = float(input("Enter mutation rate : "))
max_generations = int(input("Enter maximum number of generations: "))

best_route, min_distance = genetic_algorithm(num_nodes, distance_matrix, pop_size, mutation_rate, max_generations)
print("Best Route:", "-".join(map(str, best_route)))
print("Total Distance:", min_distance)

'''OUTPUT
Enter the distance matrix:
0 2 3 4 1
2 0 4 3 2
3 4 0 2 0
4 3 2 0 0
1 2 0 0 0
Enter population size: 100
Enter mutation rate : 0.2
Enter maximum number of generations: 100
Best Route: 0-4-1-3-2-0
Total Distance: 11
'''