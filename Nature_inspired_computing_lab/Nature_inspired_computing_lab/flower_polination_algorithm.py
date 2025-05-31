import numpy as np
import random

# Objective function
def objective_function(x):
    x1, x2, x3 = x
    return 2*x1 + 3*x2 + 4*x3 - 2

# Levy flight for global pollination
def levy_flight(Lambda, dim):
    sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size=dim)
    v = np.random.normal(0, 1, size=dim)
    return u / np.power(np.abs(v), 1 / Lambda)

# Flower Pollination Algorithm
def flower_pollination_algorithm(n, goal="min", d=3, iter_max=10, p=0.8):
    # Step 1: Initialize population
    population = np.random.uniform(-10, 10, (n, d))
    fitness = np.array([objective_function(ind) for ind in population])

    if goal == "min":
        best_index = np.argmin(fitness)
        best_fitness = np.min(fitness)
    else:
        best_index = np.argmax(fitness)
        best_fitness = np.max(fitness)

    best = population[best_index]

    print("🔹 Initial Population (Each flower has 3 variables):")
    for i, ind in enumerate(population):
        print(f"Flower {i+1}: {ind}, Fitness: {fitness[i]:.4f}")
    print(f"\n🔸 Initial Best: {best}, Fitness: {best_fitness:.4f} [{goal.upper()}IMIZATION]")
    print("\n" + "="*70 + "\n")

    for t in range(iter_max):
        print(f"🌼 Iteration {t+1}/{iter_max}")
        for i in range(n):
            if random.random() < p:
                # Global pollination
                L = levy_flight(1.5, d)
                new_solution = population[i] + L * (population[i] - best)
                method = "Global Pollination"
            else:
                # Local pollination
                j, k = np.random.choice(range(n), 2, replace=False)
                new_solution = population[i] + random.random() * (population[j] - population[k])
                method = "Local Pollination"

            new_fitness = objective_function(new_solution)
            print(f"  🌸 Flower {i+1} -> {method}")
            print(f"     Old Position: {population[i]}")
            print(f"     New Position: {new_solution}")
            print(f"     New Fitness: {new_fitness:.4f}")

            # Determine if better
            is_better = new_fitness < fitness[i] if goal == "min" else new_fitness > fitness[i]

            if is_better:
                population[i] = new_solution
                fitness[i] = new_fitness
                print("     ✅ Accepted (Improved)")
            else:
                print("     ❌ Rejected")

            # Update global best
            if (goal == "min" and new_fitness < best_fitness) or (goal == "max" and new_fitness > best_fitness):
                best = new_solution
                best_fitness = new_fitness
                print(f"     🌟 New Global Best! Fitness: {best_fitness:.4f}")

        print(f"🔹 Best after iteration {t+1}: {best}, Fitness: {best_fitness:.4f}")
        print("-" * 70)

    print(f"\n✅ Final Best Solution: {best}")
    print(f"✅ Final Best Fitness: {best_fitness:.4f}")

# === MAIN: Ask user inputs ===
if __name__ == "__main__":
    try:
        n = int(input("Enter number of flowers (e.g., 5): "))
        if n <= 0:
            print("❌ Number of flowers must be positive.")
            exit()

        mode = input("Minimization or Maximization? (min/max): ").strip().lower()
        if mode not in ["min", "max"]:
            print("❌ Please enter 'min' or 'max'")
            exit()

        flower_pollination_algorithm(n=n, goal=mode)

    except ValueError:
        print("❌ Invalid input. Please enter integers where expected.")
