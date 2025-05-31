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

def rank_based_selection(population, fitness_scores):
    sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    ranks = np.arange(1, len(population) + 1)
    probabilities = ranks / ranks.sum()
    selected = random.choices(sorted_pop, weights=probabilities, k=3)
    return [x[0] for x in selected]

def three_parent_crossover(parents):
    n = len(parents[0])
    combinations = [
        (parents[0], parents[1], parents[2]),
        (parents[1], parents[2], parents[0]),
        (parents[2], parents[0], parents[1])
    ]
    children = []
    
    for p1, p2, p3 in combinations:
        child = []
        for i in range(n):
            if p1[i] == p2[i]:
                child.append(p1[i])
            else:
                child.append(p3[i])
        children.append(child)
    
    return children

def mutate(board, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(board)), 2)
        board[i], board[j] = board[j], board[i] 
    return board

def genetic_algorithm(n, population_size, generations, mutation_rate):
    population = [random.sample(range(n), n) for _ in range(population_size)]
    
    for generation in range(generations):
        fitness_scores = [fitness(board) for board in population]
        if max(fitness_scores) == 1:
            return population[fitness_scores.index(max(fitness_scores))] 
        
        new_population = []
        for _ in range(population_size // 3):
            parents = rank_based_selection(population, fitness_scores)
            children = three_parent_crossover(parents)
            children = [mutate(child, mutation_rate) for child in children]
            new_population.extend(children)
        
        population = new_population + random.sample(population, population_size - len(new_population))
    
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
Solution found: [4, 0, 7, 5, 2, 6, 1, 3]
'''