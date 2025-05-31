import numpy as np
import random
from copy import deepcopy

def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                goal_pos = np.where(goal_state == state[i][j])
                distance += abs(i - goal_pos[0][0]) + abs(j - goal_pos[1][0])
    return distance

def get_possible_moves(state):
    moves = []
    zero_pos = np.where(state == 0)
    row, col = zero_pos[0][0], zero_pos[1][0]

    for move in ['U', 'D', 'L', 'R']:
        new_state = deepcopy(state)
        if move == 'U' and row > 0:
            new_state[row][col], new_state[row-1][col] = new_state[row-1][col], new_state[row][col]
            moves.append((new_state, move))
        elif move == 'D' and row < 2:
            new_state[row][col], new_state[row+1][col] = new_state[row+1][col], new_state[row][col]
            moves.append((new_state, move))
        elif move == 'L' and col > 0:
            new_state[row][col], new_state[row][col-1] = new_state[row][col-1], new_state[row][col]
            moves.append((new_state, move))
        elif move == 'R' and col < 2:
            new_state[row][col], new_state[row][col+1] = new_state[row][col+1], new_state[row][col]
            moves.append((new_state, move))
    return moves

def initialize_pheromones():
    return {}

def update_pheromone(pheromones, path, quality):
    for i in range(len(path) - 1):
        state_move = (tuple(path[i].flatten()), tuple(path[i+1].flatten()))
        if state_move not in pheromones:
            pheromones[state_move] = 1.0
        pheromones[state_move] += quality

def evaporate_pheromones(pheromones, evaporation_rate):
    for key in list(pheromones.keys()):
        pheromones[key] *= (1 - evaporation_rate)
        if pheromones[key] < 0.1:
            del pheromones[key]

def ant_colony_optimization(initial_state, goal_state, num_ants, evaporation_rate, alpha, beta, Q, max_iterations):
    pheromones = initialize_pheromones()
    best_path = None
    best_moves = None
    best_path_length = float('inf')

    for iteration in range(max_iterations):
        paths = []
        move_sequences = []
        path_lengths = []

        for ant in range(num_ants):
            current_state = deepcopy(initial_state)
            path = [current_state]
            moves = []
            path_length = 0

            while not np.array_equal(current_state, goal_state) and path_length < 1000:
                possible_moves = get_possible_moves(current_state)
                if not possible_moves:
                    break

                probabilities = []
                for move_state, move_dir in possible_moves:
                    state_key = (tuple(current_state.flatten()), tuple(move_state.flatten()))
                    tau = pheromones.get(state_key, 1.0) 
                    eta = 1 / (1 + manhattan_distance(move_state, goal_state))  
                    probabilities.append((tau ** alpha) * (eta ** beta))

                prob_sum = sum(probabilities)
                if prob_sum == 0:
                    probabilities = [1 / len(probabilities)] * len(probabilities)
                else:
                    probabilities = [p / prob_sum for p in probabilities]

                chosen_idx = np.random.choice(len(possible_moves), p=probabilities)
                chosen_state, chosen_dir = possible_moves[chosen_idx]
                path.append(chosen_state)
                moves.append(chosen_dir)
                current_state = chosen_state
                path_length += 1

            paths.append(path)
            move_sequences.append(moves)
            path_lengths.append(path_length)

            if np.array_equal(current_state, goal_state) and path_length < best_path_length:
                best_path = path
                best_moves = moves
                best_path_length = path_length

        evaporate_pheromones(pheromones, evaporation_rate)

        for i in range(len(paths)):
            if path_lengths[i] > 0:
                quality = Q / path_lengths[i]  
                update_pheromone(pheromones, paths[i], quality)

    return best_path, best_moves

def print_state(state):
    for row in state:
        for i, num in enumerate(row):
            print(f'{num}', end='')
            if i < 2:
                print(' ', end='')
        print()

def get_state_input(prompt):
    numbers = list(map(int, input(prompt).split()))
    return np.array(numbers).reshape(3, 3)


initial_state = get_state_input("Enter INITIAL state (9 numbers, space separated): ")
goal_state = get_state_input("Enter GOAL state (9 numbers, space separated): ")
num_ants = int(input("Enter number of ants: "))
evaporation_rate = float(input("Enter evaporation rate (0-1): "))
alpha = float(input("Enter alpha (pheromone importance): "))
beta = float(input("Enter beta (heuristic importance): "))
Q = float(input("Enter Q (pheromone deposit factor): "))
max_iterations = int(input("Enter number of iterations: "))

print("\nInitial state:")
print_state(initial_state)
print("\nGoal state:")
print_state(goal_state)

best_path, best_moves = ant_colony_optimization(
    initial_state, goal_state, num_ants, evaporation_rate, alpha, beta, Q, max_iterations
)

if best_path:
    print("\nMove sequence:")
    print(" -> ".join(best_moves))
    
    print("\nFinal solved state:")
    print_state(best_path[-1])
    
    print(f"\nPuzzle solved in {len(best_moves)} moves!")
    print(f"Manhattan Distance = {manhattan_distance(best_path[-1], goal_state)}")
else:
    print("No solution found within the given iterations.")


''' OUTPUT
Enter INITIAL state (9 numbers, space separated): 1 2 3 4 5 0 6 7 8
Enter GOAL state (9 numbers, space separated): 1 2 3 4 5 6 7 8 0
Enter number of ants: 4
Enter evaporation rate (0-1): 0.6
Enter alpha (pheromone importance): 2
Enter beta (heuristic importance): 2
Enter Q (pheromone deposit factor): 1
Enter number of iterations: 10

Initial state:
1 2 3
4 5 0
6 7 8

Goal state:
1 2 3
4 5 6
7 8 0

Move sequence:
U -> L -> D -> U -> R -> L -> L -> D -> R -> L -> D -> U -> D -> U -> D -> R -> U -> L -> U -> D 
-> U -> D -> R -> U -> D -> L -> U -> D -> D -> U -> U -> R -> R -> D -> L -> D -> L -> R -> R

Final solved state:
1 2 3
4 5 6
7 8 0

Puzzle solved in 39 moves!
Manhattan Distance = 0
'''