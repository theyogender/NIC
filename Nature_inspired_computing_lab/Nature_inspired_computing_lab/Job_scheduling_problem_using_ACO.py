import matplotlib.pyplot as plt
import random
import numpy as np

# --- Step 1: User Inputs ---
def get_user_input(prompt, default):
    user_input = input(f"{prompt} (Default: {default}): ")
    return type(default)(user_input) if user_input.strip() != "" else default

print("Enter ACO Parameters:")
NUM_ANTS = get_user_input("Number of ants", 10)
NUM_ITERATIONS = get_user_input("Number of iterations", 50)
ALPHA = get_user_input("Alpha (pheromone importance)", 1.0)
BETA = get_user_input("Beta (heuristic importance)", 2.0)
EVAPORATION_RATE = get_user_input("Evaporation rate", 0.5)
Q = get_user_input("Pheromone update constant Q", 100)

# --- Step 2: Job Definition (Job, (Machine, Time)) ---
jobs_data = [
    [(0, 3), (1, 2), (2, 2)],  # Job 0
    [(0, 2), (2, 1), (1, 4)],  # Job 1
    [(1, 4), (2, 3)]           # Job 2
]

num_jobs = len(jobs_data)
num_machines = 3
operations = [(i, j) for i in range(num_jobs) for j in range(len(jobs_data[i]))]

# Initialize Pheromone & Heuristic Matrices
pheromone = np.ones((len(operations), len(operations)))
heuristic = np.zeros((len(operations), len(operations)))

for i, (job_i, op_i) in enumerate(operations):
    for j, (job_j, op_j) in enumerate(operations):
        if (job_i != job_j) or (op_j == op_i + 1):
            machine_j, proc_time_j = jobs_data[job_j][op_j]
            heuristic[i][j] = 1.0 / proc_time_j

# --- Step 3: Compute Makespan of a Schedule ---
def compute_schedule(schedule):
    job_end_times = [0] * num_jobs
    machine_end_times = [0] * num_machines
    operation_times = {}

    for job, op in schedule:
        machine, duration = jobs_data[job][op]
        start_time = max(job_end_times[job], machine_end_times[machine])
        end_time = start_time + duration
        job_end_times[job] = end_time
        machine_end_times[machine] = end_time
        operation_times[(job, op)] = (machine, start_time, duration)

    return max(job_end_times), operation_times

# --- Step 4: Construct Ant Solution ---
def construct_solution():
    schedule = []
    current_op = [0] * num_jobs
    visited = []

    while any(current_op[job] < len(jobs_data[job]) for job in range(num_jobs)):
        possible = [(job, current_op[job]) for job in range(num_jobs) if current_op[job] < len(jobs_data[job])]
        probabilities = []
        for move in possible:
            i = operations.index(move)
            pheromone_sum = sum(
                (pheromone[prev][i] ** ALPHA) * (heuristic[prev][i] ** BETA)
                for prev in visited[-1:] if visited
            ) or 1
            probabilities.append(pheromone_sum)

        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        next_move = random.choices(possible, probabilities)[0]

        schedule.append(next_move)
        visited.append(operations.index(next_move))
        current_op[next_move[0]] += 1

    return schedule

# --- Step 5: Pheromone Update ---
def update_pheromone(all_schedules):
    global pheromone
    pheromone *= (1 - EVAPORATION_RATE)
    for schedule, makespan, _ in all_schedules:
        for i in range(len(schedule) - 1):
            a = operations.index(schedule[i])
            b = operations.index(schedule[i+1])
            pheromone[a][b] += Q / makespan

# --- Step 6: Gantt Chart Plotting ---
def plot_gantt_chart(operation_times, title):
    colors = plt.cm.get_cmap('tab20', num_jobs)
    fig, ax = plt.subplots(figsize=(10, 5))
    for (job, op), (machine, start, duration) in operation_times.items():
        ax.broken_barh([(start, duration)], (machine * 10, 9),
                       facecolors=colors(job), label=f"Job {job}" if op == 0 else "")
        ax.text(start + duration / 2, machine * 10 + 4.5,
                f"J{job}O{op}", ha='center', va='center', color='black')

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks([i * 10 + 5 for i in range(num_machines)])
    ax.set_yticklabels([f"M{i}" for i in range(num_machines)])
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- Step 7: ACO Main Loop ---
best_makespan = float('inf')
best_schedule = None
best_operation_times = {}

print("\nRunning Ant Colony Optimization...\n")

for iteration in range(NUM_ITERATIONS):
    all_schedules = []

    for ant in range(NUM_ANTS):
        schedule = construct_solution()
        makespan, operation_times = compute_schedule(schedule)
        all_schedules.append((schedule, makespan, operation_times))

        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = schedule
            best_operation_times = operation_times

    update_pheromone(all_schedules)
    print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}: Best Makespan = {best_makespan}")

# --- Step 8: Final Output ---
print("\nBest Schedule Found:\n")
for job, op in best_schedule:
    print(f"Job {job} - Operation {op}")

print(f"\nBest Makespan: {best_makespan}\n")

plot_gantt_chart(best_operation_times, title=f"Gantt Chart (Makespan = {best_makespan})")
