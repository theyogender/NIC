import random
import numpy as np
import matplotlib.pyplot as plt

def fitness(state, goal_state):
    return sum([1 for i in range(9) if state[i] == goal_state[i]])

def generate_population(size):
    population = []
    for _ in range(size):
        state = list(range(9))
        random.shuffle(state)
        population.append(state)
    return population

def select_parents(population, goal_state):
    fitness_scores = [(state, fitness(state, goal_state)) for state in population]
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    return fitness_scores[:2]  # Select top two as parents

def crossover(parent1, parent2):
    point = random.randint(1, 7)
    child1 = parent1[:point] + [x for x in parent2 if x not in parent1[:point]]
    child2 = parent2[:point] + [x for x in parent1 if x not in parent2[:point]]
    return child1, child2

def mutate(state, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(9), 2)
        state[idx1], state[idx2] = state[idx2], state[idx1]
    return state

def genetic_algorithm(initial_state, goal_state, population_size, mutation_rate, generations):
    population = generate_population(population_size)
    population.append(initial_state)
    fitness_evolution = []
    
    for gen in range(generations):
        print(f"Generation {gen+1}:")
        new_population = []
        
        for _ in range(population_size // 2):
            parents = select_parents(population, goal_state)
            child1, child2 = crossover(parents[0][0], parents[1][0])
            child1, child2 = mutate(child1, mutation_rate), mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population
        best_match = max(population, key=lambda state: fitness(state, goal_state))
        best_fitness = fitness(best_match, goal_state)
        fitness_evolution.append(best_fitness)
        print(f"Best State: {best_match} | Fitness: {best_fitness}\n")
        
        if best_fitness == 9:
            print("Goal state reached!")
            break
    
    plt.plot(range(1, len(fitness_evolution) + 1), fitness_evolution, marker='o', linestyle='-')
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness Score")
    plt.title("Evolution of Fitness Over Generations")
    plt.show()
    
    return best_match

# Get user input
initial_state = list(map(int, input("Enter initial state as 9 space-separated numbers: ").split()))
goal_state = list(map(int, input("Enter goal state as 9 space-separated numbers: ").split()))
population_size = int(input("Enter population size: "))
mutation_rate = float(input("Enter mutation rate (e.g., 0.2): "))
generations = int(input("Enter number of generations: "))

final_state = genetic_algorithm(initial_state, goal_state, population_size, mutation_rate, generations)
print("Final State:", final_state)
