import numpy as np
import random
from collections import defaultdict

class Operation:
    def __init__(self, job_id, op_id, machine, processing_time):
        self.job_id = job_id
        self.op_id = op_id
        self.machine = machine
        self.processing_time = processing_time
        self.start_time = None
        self.end_time = None

class ACO_JSSP:
    def __init__(self, num_ants, evaporation_rate, alpha, beta, q, iterations):
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.iterations = iterations
        
    def setup_problem(self, num_jobs, num_machines, jobs_data):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.jobs = [[] for _ in range(num_jobs + 1)]
        
        for job_id in range(1, num_jobs + 1):
            operations = jobs_data[job_id]
            for op_id in range(1, len(operations) + 1):
                machine, processing_time = operations[op_id - 1]
                self.jobs[job_id].append(Operation(job_id, op_id, machine, processing_time))
        
        total_ops = sum(len(job) for job in self.jobs if job)
        self.pheromone = np.ones((total_ops + 1, total_ops + 1)) / total_ops
        self.best_schedule = None
        self.best_makespan = float('inf')
    
    def run(self):
        for _ in range(self.iterations):
            ant_schedules = []
            for _ in range(self.num_ants):
                schedule = self.construct_schedule()
                makespan = self.calculate_makespan(schedule)
                ant_schedules.append((schedule, makespan))
                if makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_schedule = schedule
            self.update_pheromone(ant_schedules)
        return self.best_schedule, self.best_makespan
    
    def construct_schedule(self):
        scheduled_ops = set()
        machine_available = [0] * (self.num_machines + 1)
        job_next_op = [1] * (self.num_jobs + 1)
        schedule = []
        
        all_ops = []
        for job in self.jobs[1:]:
            all_ops.extend(job)
        
        while len(scheduled_ops) < len(all_ops):
            candidates = []
            for job_id in range(1, self.num_jobs + 1):
                if job_next_op[job_id] <= len(self.jobs[job_id]):
                    op = self.jobs[job_id][job_next_op[job_id] - 1]
                    if op not in scheduled_ops and (op.op_id == 1 or self.jobs[job_id][op.op_id - 2] in scheduled_ops):
                        candidates.append(op)
            if not candidates:
                break
            next_op = self.select_next_operation(candidates, all_ops, scheduled_ops)
            prev_op_end = 0
            if next_op.op_id > 1:
                prev_op = self.jobs[next_op.job_id][next_op.op_id - 2]
                prev_op_end = prev_op.end_time
            start_time = max(machine_available[next_op.machine], prev_op_end)
            next_op.start_time = start_time
            next_op.end_time = start_time + next_op.processing_time
            machine_available[next_op.machine] = next_op.end_time
            schedule.append(next_op)
            scheduled_ops.add(next_op)
            job_next_op[next_op.job_id] += 1
        return schedule
    
    def select_next_operation(self, candidates, all_ops, scheduled_ops):
        op_indices = {op: idx + 1 for idx, op in enumerate(all_ops)}
        probabilities = []
        for op in candidates:
            pheromone = np.mean([self.pheromone[op_indices[op]][op_indices[o]] for o in all_ops if o not in scheduled_ops and o != op] or [1])
            heuristic = 1 / op.processing_time
            probabilities.append((pheromone ** self.alpha) * (heuristic ** self.beta))
        total = sum(probabilities)
        probabilities = [p/total if total > 0 else 1/len(candidates) for p in probabilities]
        return random.choices(candidates, weights=probabilities, k=1)[0]
    
    def update_pheromone(self, ant_schedules):
        self.pheromone *= (1 - self.evaporation_rate)
        all_ops = []
        for job in self.jobs[1:]:
            all_ops.extend(job)
        op_indices = {op: idx + 1 for idx, op in enumerate(all_ops)}
        for schedule, makespan in ant_schedules:
            pheromone_to_add = self.q / makespan
            for i in range(len(schedule) - 1):
                curr, next_op = op_indices[schedule[i]], op_indices[schedule[i + 1]]
                self.pheromone[curr][next_op] += pheromone_to_add
                self.pheromone[next_op][curr] += pheromone_to_add
    
    def calculate_makespan(self, schedule):
        return max(op.end_time for op in schedule) if schedule else float('inf')

num_jobs = int(input("Enter number of jobs: "))
num_machines = int(input("Enter number of machines: "))
jobs_data = defaultdict(list)

for job_id in range(1, num_jobs + 1):
    num_operations = int(input(f"\nEnter number of operations for job {job_id}: "))
    for op_num in range(1, num_operations + 1):
        machine = int(input(f"Enter machine for operation {op_num} (1-{num_machines}): "))
        processing_time = int(input(f"Enter time for operation {op_num}: "))
        jobs_data[job_id].append((machine, processing_time))

num_ants = int(input("Enter number of ants: "))
evaporation_rate = float(input("Enter evaporation rate (0-1): "))
alpha = float(input("Enter alpha (pheromone importance): "))
beta = float(input("Enter beta (heuristic importance): "))
q = float(input("Enter Q (pheromone deposit factor): "))
iterations = int(input("Enter number of iterations: "))

aco = ACO_JSSP( num_ants, evaporation_rate, alpha, beta, q, iterations )
aco.setup_problem(num_jobs, num_machines, jobs_data)
best_schedule, best_makespan = aco.run()

print(f"\nOptimal Path: {' -> '.join(f'J{op.job_id}O{op.op_id}' for op in best_schedule)}")
print(f"Total Time: {best_makespan}")

''' OUTPUT
Enter number of jobs: 3
Enter number of machines: 3

Enter number of operations for job 1: 3
Enter machine for operation 1 (1-3): 2
Enter time for operation 1: 3
Enter machine for operation 2 (1-3): 1
Enter time for operation 2: 2
Enter machine for operation 3 (1-3): 3
Enter time for operation 3: 4

Enter number of operations for job 2: 3
Enter machine for operation 1 (1-3): 1
Enter time for operation 1: 2
Enter machine for operation 2 (1-3): 2
Enter time for operation 2: 3
Enter machine for operation 3 (1-3): 3
Enter time for operation 3: 1

Enter number of operations for job 3: 3
Enter machine for operation 1 (1-3): 2
Enter time for operation 1: 1
Enter machine for operation 2 (1-3): 3
Enter time for operation 2: 2
Enter machine for operation 3 (1-3): 1
Enter time for operation 3: 1
Enter number of ants: 3
Enter evaporation rate (0-1): 0.5
Enter alpha (pheromone importance): 1
Enter beta (heuristic importance): 1
Enter Q (pheromone deposit factor): 1
Enter number of iterations: 100

Optimal Path: J1O1 -> J1O2 -> J2O1 -> J3O1 -> J3O2 -> J3O3 -> J1O3 -> J2O2 -> J2O3
Total Time: 11
'''