import numpy as np

# Objective function (Example: Sphere function)
def objective_function(x):
    return sum((x**2)+(x*11)+11)

# PSO parameters
num_particles = 4
num_dimensions = 2
num_iterations = 5
w = 0.7             # Inertia weight
c1 = 1.5            # Cognitive (particle best) coefficient
c2 = 1.5            # Social (global best) coefficient

# Initialize particle positions and velocities
positions = np.random.uniform(-10, 10, (num_particles, num_dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))

# Initialize personal bests and global best
personal_best_positions = positions.copy()
personal_best_scores = np.array([objective_function(p) for p in positions])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = min(personal_best_scores)

# PSO main loop
for iteration in range(num_iterations):
    for i in range(num_particles):
        # Evaluate the objective function
        fitness = objective_function(positions[i])

        # Update personal best
        if fitness < personal_best_scores[i]:
            personal_best_scores[i] = fitness
            personal_best_positions[i] = positions[i]

        # Update global best
        if fitness < global_best_score:
            global_best_score = fitness
            global_best_position = positions[i]

        # Update velocity using the PSO equation
        r1, r2 = np.random.rand(num_dimensions), np.random.rand(num_dimensions)
        velocities[i] = (w * velocities[i] +
                          c1 * r1 * (personal_best_positions[i] - positions[i]) +
                          c2 * r2 * (global_best_position - positions[i]))

        # Update position
        positions[i] = positions[i] + velocities[i]

    print(f"Iteration {iteration+1}/{num_iterations}, Best Score: {global_best_score}")

print("Optimal Solution:", global_best_position)
print("Optimal Objective Value:", global_best_score)
