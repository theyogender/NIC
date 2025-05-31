import random
import numpy as np

def fitness(board):
    n = len(board)
    clashes = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(board[i] - board[j]) == j - i:  
                clashes += 1
    return 1 / (1 + clashes)  

def tournament_selection(population, fitness_scores, tournament_size=3):
    selected = random.sample(list(zip(population, fitness_scores)), tournament_size)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]  

def uniform_crossover(parent1, parent2):
    n = len(parent1)
    mask = [random.randint(0, 1) for _ in range(n)]  
    child = [parent1[i] if mask[i] == 1 else parent2[i] for i in range(n)]
    return child

def mutate(board, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(board)), 2)
        board[i], board[j] = board[j], board[i]  
    return board

def genetic_algorithm(n, population_size, generations, mutation_rate, elitism_ratio=0.1):
    population = [random.sample(range(n), n) for _ in range(population_size)]
    
    for generation in range(generations):
        fitness_scores = [fitness(board) for board in population]
        sorted_pop = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
        
        if max(fitness_scores) == 1:
            return population[fitness_scores.index(max(fitness_scores))]  
        
        elitism_count = int(elitism_ratio * population_size)
        new_population = sorted_pop[:elitism_count]  
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child = uniform_crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
    
    return None  

n = int(input("Enter the number of queens: "))
population_size = int(input("Enter the population size: "))
mutation_rate = float(input("Enter the mutation rate (0-1): "))
generations = int(input("Enter the maximum number of iterations: "))

solution = genetic_algorithm(n, population_size, generations, mutation_rate)
if solution:
    print("Solution found:", solution)
else:
    print("No solution found within the given generations.")

'''OUTPUT
Enter the number of queens: 8
Enter the population size: 500
Enter the mutation rate (0-1): 0.2
Enter the maximum number of iterations: 500
Solution found: [1, 6, 2, 7, 1, 4, 0, 5]
'''