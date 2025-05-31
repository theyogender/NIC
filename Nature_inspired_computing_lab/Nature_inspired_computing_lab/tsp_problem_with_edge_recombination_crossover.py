import numpy as np
import random
import matplotlib.pyplot as plt

def input_city_data():
    num_cities = int(input("Enter number of cities: "))
    cities = []
    for i in range(num_cities):
        city_name = input(f"Enter name of city {i+1}: ")
        cities.append(city_name)

    distance_matrix = np.full((num_cities, num_cities), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = input(f"Enter distance between {cities[i]} and {cities[j]} (or press Enter if no path): ")
            if distance.strip():
                distance_matrix[i, j] = float(distance)
                distance_matrix[j, i] = float(distance)

    starting_city = input("Enter the starting city: ")
    start_index = cities.index(starting_city)

    return num_cities, cities, distance_matrix, start_index

def initial_population(num_cities, population_size):
    return [random.sample(range(num_cities), num_cities) for _ in range(population_size)]

def calculate_fitness(path, distance_matrix):
    return sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)) + distance_matrix[path[-1], path[0]]

def edge_recombination_crossover(parent1, parent2):
    def build_edge_map(parent1, parent2):
        edges = {i: set() for i in parent1}
        for p in [parent1, parent2]:
            for i in range(len(p)):
                left = p[i - 1]
                right = p[(i + 1) % len(p)]
                edges[p[i]].update([left, right])
        return edges

    edges = build_edge_map(parent1, parent2)
    current = random.choice(parent1)
    child = [current]

    while len(child) < len(parent1):
        for neighbors in edges.values():
            neighbors.discard(current)
        next_city = min(edges[current], key=lambda x: len(edges[x]), default=None)
        if next_city is None:
            next_city = random.choice(list(set(parent1) - set(child)))
        child.append(next_city)
        current = next_city

    return child

def mutate(path):
    a, b = random.sample(range(len(path)), 2)
    path[a], path[b] = path[b], path[a]

def genetic_algorithm(num_cities, distance_matrix, population_size=100, generations=500, mutation_rate=0.1):
    population = initial_population(num_cities, population_size)

    for gen in range(generations):
        population.sort(key=lambda path: calculate_fitness(path, distance_matrix))
        best_path = population[0]
        best_distance = calculate_fitness(best_path, distance_matrix)
        print(f"Generation {gen+1}: Best Path: {' -> '.join(str(x) for x in best_path)}, Distance: {best_distance}")

        next_population = population[:10]

        while len(next_population) < population_size:
            parent1, parent2 = random.choices(population[:50], k=2)
            child = edge_recombination_crossover(parent1, parent2)
            if random.random() < mutation_rate:
                mutate(child)
            next_population.append(child)
        population = next_population

    best_path = min(population, key=lambda path: calculate_fitness(path, distance_matrix))
    best_distance = calculate_fitness(best_path, distance_matrix)
    return best_path, best_distance

def plot_path(best_path, cities):
    points = np.random.rand(len(best_path), 2) * 100
    plt.figure()
    for i in range(len(best_path)):
        plt.plot([points[best_path[i], 0], points[best_path[(i + 1) % len(best_path)], 0]],
                 [points[best_path[i], 1], points[best_path[(i + 1) % len(best_path)], 1]], 'bo-')
        plt.text(points[best_path[i], 0], points[best_path[i], 1], cities[best_path[i]], fontsize=12)
    plt.title('Best Path')
    plt.show()

def main():
    num_cities, cities, distance_matrix, start_index = input_city_data()
    population_size = int(input("Enter population size: "))
    generations = int(input("Enter number of generations: "))
    mutation_rate = float(input("Enter mutation rate (e.g., 0.1 for 10%): "))

    best_path, best_distance = genetic_algorithm(num_cities, distance_matrix, population_size, generations, mutation_rate)
    print("Best Path:", ' -> '.join(cities[i] for i in best_path))
    print("Best Distance:", best_distance)
    plot_path(best_path, cities)

if __name__ == "__main__":
    main()
