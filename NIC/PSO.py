import numpy as np

def objective_function(x):
    return 1 + 2*x[0] + 2*(x[1] + 1) + 3*(x[2] - 1)

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)])
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def update_velocity(self, global_best_position, inertia, cognitive, social):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_component = cognitive * r1 * (self.best_position - self.position)
        social_component = social * r2 * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_component + social_component

    def update_position(self, bounds):
        self.position += self.velocity
        for i in range(len(self.position)):
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])

def pso(dim, bounds, num_particles, max_iter, inertia, cognitive, social, opt_type):
    swarm = [Particle(dim, bounds) for _ in range(num_particles)]
    global_best_position = np.copy(swarm[0].position)
    global_best_score = objective_function(global_best_position)

    for _ in range(max_iter):
        for particle in swarm:
            score = objective_function(particle.position)

            if (opt_type == 0 and score < particle.best_score) or (opt_type == 1 and score > particle.best_score):
                particle.best_score = score
                particle.best_position = np.copy(particle.position)

            if (opt_type == 0 and score < global_best_score) or (opt_type == 1 and score > global_best_score):
                global_best_score = score
                global_best_position = np.copy(particle.position)

        for particle in swarm:
            particle.update_velocity(global_best_position, inertia, cognitive, social)
            particle.update_position(bounds)

    return global_best_position, global_best_score

min_values = list(map(float, input("Enter the minimum values separated by space: ").split()))
max_values = list(map(float, input("Enter the maximum values separated by space: ").split()))
if len(min_values) != 3 or len(max_values) != 3:
    raise ValueError("This objective function requires exactly 3 variables.")

dim = 3
bounds = list(zip(min_values, max_values))

num_particles = int(input("Enter number of particles (N): "))
max_iter = int(input("Enter maximum number of iterations (max_iter): "))
inertia = float(input("Enter inertia weight (w): "))
cognitive = float(input("Enter cognition coefficient (C1): "))
social = float(input("Enter social coefficient (C2): "))
opt_type = int(input("Enter 0 for minimization or 1 for maximization: "))

best_position, best_score = pso(dim, bounds, num_particles, max_iter, inertia, cognitive, social, opt_type)
print("\nBest particle position:", best_position)
print("Best score:", best_score)

'''OUTPUT
Enter the minimum values separated by space: 0 0 0
Enter the maximum values separated by space: 10 10 10
Enter number of particles (N): 5
Enter maximum number of iterations (max_iter): 100
Enter inertia weight (w): 0.5
Enter cognition coefficient (C1): 1
Enter social coefficient (C2): 1
Enter 0 for minimization or 1 for maximization: 0

Best particle position: [1.01949536 3.65461866 0.        ]
Best score: 9.348228045493853
'''