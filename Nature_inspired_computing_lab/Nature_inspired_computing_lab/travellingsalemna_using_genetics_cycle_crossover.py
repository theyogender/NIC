import random
import numpy as np

def get_user_input():
    while True:
        try:
            num_cities = int(input("Enter the number of cities: "))
            if num_cities < 2:
                print("Number of cities must be at least 2.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")
    
    cities = {}
    for i in range(num_cities):
        while True:
            try:
                x, y = map(int, input(f"Enter coordinates for city {i} (x y): ").split())
                cities[i] = (x, y)
                break
            except ValueError:
                print("Invalid input. Please enter two integers separated by space.")
    
    cost_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            while True:
                try:
                    cost = int(input(f"Enter cost between city {i} and city {j}: "))
                    cost_matrix[i][j] = cost_matrix[j][i] = cost
                    break
                except ValueError:
                    print("Invalid input. Please enter an integer.")
    
    while True:
        try:
            pop_size = int(input("Enter population size: "))
            generations = int(input("Enter number of generations: "))
            mutation_rate = float(input("Enter mutation rate (e.g., 0.01): "))
            if pop_size <= 0 or generations <= 0 or not (0 <= mutation_rate <= 1):
                print("Invalid values. Ensure positive population size, generations, and mutation rate between 0 and 1.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
    
    return num_cities, cities, cost_matrix, pop_size, generations, mutation_rate

def total_distance(path, cost_matrix):
    return sum(cost_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)) + cost_matrix[path[-1]][path[0]]

def initial_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

def selection(population, cost_matrix):
    fitness = [(total_distance(ind, cost_matrix), ind) for ind in population]
    fitness.sort()
    return [ind for _, ind in fitness[: len(population) // 2]]

def cyclic_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    index = 0
    
    while child[index] == -1:
        child[index] = parent1[index]
        index = parent1.index(parent2[index])
    
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]
    
    return child

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def genetic_algorithm():
    num_cities, cities, cost_matrix, pop_size, generations, mutation_rate = get_user_input()
    population = initial_population(pop_size, num_cities)
    
    for gen in range(generations):
        selected = selection(population, cost_matrix)
        new_population = []
        
        while len(new_population) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child = cyclic_crossover(p1, p2)
            new_population.append(mutate(child, mutation_rate))
        
        population = new_population
        best_gen = min(population, key=lambda ind: total_distance(ind, cost_matrix))
        best_distance = total_distance(best_gen, cost_matrix)
        print(f"Generation {gen + 1}: Best Path {best_gen}, Distance {best_distance}")
    
    best = min(population, key=lambda ind: total_distance(ind, cost_matrix))
    return best, total_distance(best, cost_matrix), cities

if __name__ == "__main__":
    best_path, best_distance, city_coords = genetic_algorithm()
    
    print("Best Path:", best_path)
    print("Best Distance:", best_distance)
