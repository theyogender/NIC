import random

class NQueensGA:
    def __init__(self, n, population_size=100, generations=1000, mutation_rate=0.05, tournament_size=5, elitism=True):
        self.n = n
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population = [self.random_solution() for _ in range(population_size)]

    def random_solution(self):
        return random.sample(range(self.n), self.n)

    def fitness(self, solution):
        attacking_pairs = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(solution[i] - solution[j]) == abs(i - j):
                    attacking_pairs += 1
        return attacking_pairs

    def tournament_selection(self):
        competitors = random.sample(self.population, self.tournament_size)
        return min(competitors, key=self.fitness)

    def uniform_crossover(self, parent1, parent2):
        child = [-1] * self.n
        for i in range(self.n):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        # Fix invalid child by making it a valid permutation
        missing_values = list(set(range(self.n)) - set(child))
        for i in range(self.n):
            if child[i] == -1:
                child[i] = missing_values.pop()
        return child

    def mutate(self, solution):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.n), 2)
            solution[i], solution[j] = solution[j], solution[i]

    def evolve(self):
        for generation in range(self.generations):
            new_population = []
            
            # Elitism: Keep the best solution
            if self.elitism:
                best_solution = min(self.population, key=self.fitness)
                new_population.append(best_solution)
                print(f"Generation {generation+1}: Elitism Maintained: {best_solution}, Fitness: {self.fitness(best_solution)}")
            
            # Generate offspring using selection, crossover, and mutation
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child = self.uniform_crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
                print(f"Generation {generation+1}: Parents: {parent1}, {parent2} => Child: {child}")

            self.population = new_population
            best_solution = min(self.population, key=self.fitness)
            print(f"Generation {generation+1}: Best Solution = {best_solution}, Fitness = {self.fitness(best_solution)}")

            if self.fitness(best_solution) == 0:
                print("\nSolution Found!")
                print(best_solution)
                return

        print("\nNo perfect solution found. Best attempt:")
        print(min(self.population, key=self.fitness))


if __name__ == "__main__":
    n = int(input("Enter number of queens: "))
    ga = NQueensGA(n)
    ga.evolve()
