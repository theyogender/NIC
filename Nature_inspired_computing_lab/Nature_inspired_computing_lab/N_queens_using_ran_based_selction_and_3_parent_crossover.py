import random

def generate_chromosome(n):
    return [random.randint(1, n) for _ in range(n)]

def fitness(chromosome):
    n = len(chromosome)
    attacking_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == abs(i - j):
                attacking_pairs += 1
    return attacking_pairs

def rank_based_selection(population):
    population.sort(key=lambda x: fitness(x))
    ranks = list(range(1, len(population) + 1))
    total_rank = sum(ranks)
    probabilities = [rank / total_rank for rank in ranks]
    return random.choices(population, probabilities, k=3)

def three_parent_crossover(parents):
    n = len(parents[0])
    offspring = []
    for i in range(n):
        offspring.append(random.choice([parents[0][i], parents[1][i], parents[2][i]]))
    return offspring if len(set(offspring)) == n else generate_chromosome(n)

def mutate(chromosome, mutation_rate=0.1):
    if random.random() < mutation_rate:
        n = len(chromosome)
        i, j = random.sample(range(n), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

def genetic_algorithm(n, population_size=100, generations=1000, mutation_rate=0.1):
    population = [generate_chromosome(n) for _ in range(population_size)]

    for generation in range(generations):
        population.sort(key=lambda x: fitness(x))
        print(f"Generation {generation}: Best Fitness = {fitness(population[0])}, Best Position = {population[0]}")
        
        if fitness(population[0]) == 0:
            print(f"Solution found in generation {generation}")
            return population[0]

        new_population = []
        while len(new_population) < population_size:
            parents = rank_based_selection(population)
            offspring = three_parent_crossover(parents)
            mutate(offspring, mutation_rate)
            new_population.append(offspring)
        
        population = new_population

    print("No solution found")
    return None

n = int(input("Enter the number of queens: "))
solution = genetic_algorithm(n)
if solution:
    print("Solution:", solution) 
else: 
    print("Try adjusting parameters or increasing generations.")
