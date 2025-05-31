import numpy as np
import random
import matplotlib.pyplot as plt

# Function to generate a distance matrix for an undirected graph
def generate_distance_matrix(num_cities, city_names, path):
    distance_matrix = np.full((num_cities, num_cities), np.inf)
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                if city_names[j] in path.get(city_names[i], {}):
                    distance_matrix[i][j] = path[city_names[i]][city_names[j]]
                # Since the graph is undirected, set both directions
                if city_names[i] in path.get(city_names[j], {}):
                    distance_matrix[j][i] = path[city_names[j]][city_names[i]]
    return distance_matrix

# Function to create the initial population with a fixed start city
def initial_population(num_cities, population_size, start_city_index):
    cities = list(range(num_cities))
    cities.remove(start_city_index)
    return [np.concatenate(([start_city_index], np.random.permutation(cities))) for _ in range(population_size)]

# Fitness function to evaluate the total distance of a path
def calculate_fitness(individual, distance_matrix):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += distance_matrix[individual[i], individual[i + 1]]
    total_distance += distance_matrix[individual[-1], individual[0]]  # Return to start city
    return 1 / total_distance if total_distance != np.inf else 0

# Function to select parents using fitness-proportional selection
def select_parents(population, fitnesses):
    if np.sum(fitnesses) == 0:
        return random.choice(population), random.choice(population)
    probabilities = fitnesses / np.sum(fitnesses)
    return population[np.random.choice(len(population), p=probabilities)], population[np.random.choice(len(population), p=probabilities)]

# Partially Mapped Crossover (PMX) operator for two parents
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    p1, p2 = sorted(random.sample(range(size), 2))

    child1, child2 = np.full(size, -1), np.full(size, -1)
    child1[p1:p2+1], child2[p1:p2+1] = parent1[p1:p2+1], parent2[p1:p2+1]

    def pmx_mapping(child, p1, p2, parent_a, parent_b):
        for i in range(p1, p2+1):
            while parent_b[i] in child:
                parent_b[i] = parent_a[np.where(parent_b == parent_b[i])[0][0]]
        child[child == -1] = parent_b[child == -1]

    pmx_mapping(child1, p1, p2, parent1, parent2)
    pmx_mapping(child2, p1, p2, parent2, parent1)

    return child1, child2

# Mutation operator that swaps two cities in the path
def mutate(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        a, b = np.random.choice(len(individual), 2, replace=False)
        individual[a], individual[b] = individual[b], individual[a]

# Main function implementing the genetic algorithm
def genetic_algorithm(num_cities, population_size, generations, mutation_rate, city_names, path, start_city):
    start_city_index = city_names.index(start_city)
    distance_matrix = generate_distance_matrix(num_cities, city_names, path)
    population = initial_population(num_cities, population_size, start_city_index)
    best_fitness = []
    best_individual = None
    convergence_counter = 0
    previous_best_fitness = -np.inf

    # Genetic algorithm loop
    for gen in range(generations):
        fitnesses = np.array([calculate_fitness(ind, distance_matrix) for ind in population])
        current_best_fitness = np.max(fitnesses)
        best_fitness.append(current_best_fitness)
        best_individual = population[np.argmax(fitnesses)]

        # Print the best path and fitness of the current generation
        print(f"Generation {gen+1}, Best Fitness: {best_fitness[-1]}")
        print("Current Best Path:", ' -> '.join(city_names[i] for i in best_individual))
        print("Current Best Path Distance:", 1 / best_fitness[-1])  # Distance is the inverse of fitness

        # Check for convergence (no improvement in best fitness for 50 generations)
        if current_best_fitness == previous_best_fitness:
            convergence_counter += 1
        else:
            convergence_counter = 0
        previous_best_fitness = current_best_fitness

        if convergence_counter > 50:
            print("Convergence reached, terminating early.")
            break

        # Create new population using crossover and mutation
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = pmx_crossover(parent1, parent2)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    # Final output of the best individual after all generations
    final_best_individual = population[np.argmax([calculate_fitness(ind, distance_matrix) for ind in population])]
    print("Optimal Path:", ' -> '.join(city_names[i] for i in final_best_individual))
    print("Optimal Path Distance:", 1 / calculate_fitness(final_best_individual, distance_matrix))  # Inverse of fitness
    plot_fitness(best_fitness)

# Function to plot the fitness evolution over generations
def plot_fitness(fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title('Fitness Over Generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    num_cities = int(input('Enter number of cities: '))
    city_names = []
    for i in range(num_cities):
        city_names.append(input(f'Enter name of city {i+1}: '))

    path = {}
    for city in city_names:
        path[city] = {}
        for other_city in city_names:
            if city != other_city:
                distance = input(f'Enter distance from {city} to {other_city} (or press Enter if no path): ')
                if distance.strip():
                    path[city][other_city] = float(distance)
                    # Ensure reverse direction is also initialized if not present
                    if other_city not in path:
                        path[other_city] = {}
                    path[other_city][city] = float(distance)

    start_city = input('Enter the starting city: ')

    population_size = int(input('Enter population size: '))
    generations = int(input('Enter number of generations: '))
    mutation_rate = float(input('Enter mutation rate (e.g., 0.1 for 10%): '))

    genetic_algorithm(num_cities, population_size, generations, mutation_rate, city_names, path, start_city)
