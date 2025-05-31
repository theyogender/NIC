import numpy as np
import random

class NQueensACO:
    def __init__(self, n, num_ants, alpha=1.0, beta=2.0, evaporation_rate=0.1, max_iterations=1000):
        self.n = n
        self.num_ants = num_ants
        self.alpha = alpha  # pheromone influence
        self.beta = beta    # heuristic influence
        self.evaporation_rate = evaporation_rate
        self.max_iterations = max_iterations

        # Initialize pheromone matrix (n rows x n columns)
        self.pheromone = np.ones((n, n)) * 0.1

    def heuristic(self, row, col, current_solution):
        conflicts = 0
        for r, c in enumerate(current_solution):
            if c == col or abs(r - row) == abs(c - col):
                conflicts += 1
        return 1.0 / (1 + conflicts)

    def fitness(self, solution):
        # Total non-attacking pairs
        n = len(solution)
        non_attacking = 0
        for i in range(n):
            for j in range(i + 1, n):
                if solution[i] != solution[j] and abs(i - j) != abs(solution[i] - solution[j]):
                    non_attacking += 1
        return non_attacking

    def is_valid(self, solution):
        return self.fitness(solution) == self.n * (self.n - 1) // 2

    def construct_solution(self, ant_id):
        solution = []
        path = []  # stores (row, col) pairs
        available_columns = list(range(self.n))

        for row in range(self.n):
            probs = []
            for col in available_columns:
                h = self.heuristic(row, col, solution)
                tau = self.pheromone[row][col]
                prob = (tau ** self.alpha) * (h ** self.beta)
                probs.append(prob)

                print(f"Ant {ant_id} | Row {row}, Col {col} | Pheromone: {tau:.4f}, Heuristic: {h:.4f}, Score: {prob:.4f}")

            total = sum(probs)
            probs = [p / total for p in probs]

            selected_col = random.choices(available_columns, weights=probs, k=1)[0]
            print(f"Ant {ant_id} selected: (Row {row}, Col {selected_col})")
            path.append((row, selected_col))
            solution.append(selected_col)
            available_columns.remove(selected_col)

        return solution, path

    def update_pheromones(self, all_solutions, fitnesses):
        # Evaporate pheromones
        self.pheromone *= (1 - self.evaporation_rate)

        # Deposit pheromones
        for solution, fit in zip(all_solutions, fitnesses):
            if fit == 0:
                continue
            for row, col in enumerate(solution):
                self.pheromone[row][col] += fit / 100.0  # scale update

    def solve(self):
        for iteration in range(self.max_iterations):
            print(f"\n==================== Iteration {iteration + 1} ====================")
            all_solutions = []
            fitnesses = []

            for ant in range(self.num_ants):
                solution, path = self.construct_solution(ant)
                fit = self.fitness(solution)
                print(f"Ant {ant} path: {path}")
                print(f"Ant {ant} solution: {solution} | Fitness: {fit}")
                all_solutions.append(solution)
                fitnesses.append(fit)

                if self.is_valid(solution):
                    print(f"\n✅ VALID SOLUTION FOUND by Ant {ant} at Iteration {iteration + 1}!")
                    print(f"Path taken by Ant {ant}: {path}")
                    return solution, path

            self.update_pheromones(all_solutions, fitnesses)

        print("\n❌ No valid solution found within the maximum iterations.")
        return None, None

# === User Input ===
n = int(input("Enter the number of queens (n): "))
num_ants = int(input("Enter the number of ants: "))

aco = NQueensACO(n=n, num_ants=num_ants)
solution, path = aco.solve()

if solution:
    print("\nFinal Valid Board Configuration (row: column):")
    for row, col in enumerate(solution):
        print(f"Row {row} -> Column {col}")

    print("\nPath used by the ant to build solution:")
    print(path)

    # Optional: print chessboard
    print("\nChessboard View:")
    for row in range(n):
        line = ""
        for col in range(n):
            if solution[row] == col:
                line += " Q "
            else:
                line += " . "
        print(line)
else:
    print("No valid solution was found.")
