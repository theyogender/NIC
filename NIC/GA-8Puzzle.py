import random

N = 3 

def fitness(chromosome, target):
    return sum([1 for i in range(len(chromosome)) if chromosome[i] == target[i]])

def generate_initial_state():
    state = list(range(9))
    random.shuffle(state)
    return state

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def create_initial_population(pop_size):
    return [generate_initial_state() for _ in range(pop_size)]

def genetic_algorithm(initial_state, target_state, pop_size, mutation_rate, max_generations):
    population = [initial_state] + create_initial_population(pop_size - 1) 
    generation = 0
    while generation < max_generations:
        population.sort(key=lambda x: fitness(x, target_state), reverse=True)

        if fitness(population[0], target_state) == len(target_state):
            print(f"Solution found in generation {generation}")
            print(population[0])
            return population[0]

        parents = population[:pop_size // 2]

        next_generation = []

        for _ in range(pop_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            next_generation.append(child)

        next_generation = [mutate(child, mutation_rate) for child in next_generation]

        population = parents + next_generation
        generation += 1

    print("Solution not found within maximum generations")
    return None

def get_user_input():
    print("Enter the target state (space-separated 9 values including 0 for the empty space):")
    target_state = list(map(int, input().split()))
    
    if len(target_state) != 9 or sorted(target_state) != list(range(9)):
        print("Invalid target state. Please provide exactly 9 integers between 0 and 8.")
        return None

    print("Enter the initial state (space-separated 9 values including 0 for the empty space):")
    initial_state = list(map(int, input().split()))
    
    if len(initial_state) != 9 or sorted(initial_state) != list(range(9)):
        print("Invalid initial state. Please provide exactly 9 integers between 0 and 8.")
        return None
    
    pop_size = int(input("Enter the population size (e.g., 100):"))
    mutation_rate = float(input("Enter the mutation rate (e.g., 0.1):"))
    max_generations = int(input("Enter the maximum number of generations (e.g., 1000):"))
    
    return initial_state, target_state, pop_size, mutation_rate, max_generations

user_input = get_user_input()
if user_input:
    initial_state, target_state, pop_size, mutation_rate, max_generations = user_input
    solution = genetic_algorithm(initial_state, target_state, pop_size, mutation_rate, max_generations)
    if solution:
        print(f"Solution: {solution}")
    else:
        print("No solution found.")

'''OUTPUT
Enter the target state (space-separated 9 values including 0 for the empty space):
1 2 3 4 5 0 6 7 8
Enter the initial state (space-separated 9 values including 0 for the empty space):
1 2 3 4 5 6 7 8 0
Enter the population size (e.g., 100):100
Enter the mutation rate (e.g., 0.1):0.2
Enter the maximum number of generations (e.g., 1000):100
Solution found in generation 8
[1, 2, 3, 4, 5, 0, 6, 7, 8]
Solution: [1, 2, 3, 4, 5, 0, 6, 7, 8]
'''