import numpy as np
import random

def custom_objective(variables_values=[0, 0, 0]):
    return 1 + 2*variables_values[0] + 2*(variables_values[1] + 1) + 3*(variables_values[2] - 1)

def initial_position(flowers=3, min_values=[0, 5, 10], max_values=[20, 20, 50], target_function=custom_objective):
    position = np.zeros((flowers, len(min_values) + 1))
    for i in range(flowers):
        for j in range(len(min_values)):
            position[i, j] = random.uniform(min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, :-1])
    return position

def pollination_global(position, best_global, flower, gama=0.5, lamb=0.9, min_values=[0, 5, 10], max_values=[20, 20, 50], target_function=custom_objective):
    x = np.copy(best_global)
    for j in range(len(min_values)):
        x[j] = np.clip(position[flower, j] + gama * lamb * (position[flower, j] - best_global[j]), min_values[j], max_values[j])
    x[-1] = target_function(x[:-1])
    return x

def pollination_local(position, best_global, flower, nb_flower_1, nb_flower_2, min_values=[0, 5, 10], max_values=[20, 20, 50], r=0.2, target_function=custom_objective):
    x = np.copy(best_global)
    for j in range(len(min_values)):
        x[j] = np.clip(position[flower, j] + r * (position[nb_flower_1, j] - position[nb_flower_2, j]), min_values[j], max_values[j])
    x[-1] = target_function(x[:-1])
    return x

def flower_pollination_algorithm(flowers=20, min_values=[0, 5, 10], max_values=[20, 20, 50], iterations=200, gama=0.1, lamb=0.9, p=0.8, r=0.2, target_function=custom_objective, maximize=False):
    count = 0
    position = initial_position(flowers, min_values, max_values, target_function)

    if maximize:
        best_global = np.copy(position[position[:, -1].argsort()][-1, :])
    else:
        best_global = np.copy(position[position[:, -1].argsort()][0, :])

    global_best_value = best_global[-1]

    while count <= iterations:
        for i in range(position.shape[0]):
            nb_flower_1 = np.random.randint(position.shape[0])
            nb_flower_2 = np.random.randint(position.shape[0])
            while nb_flower_1 == nb_flower_2:
                nb_flower_1 = np.random.randint(position.shape[0])

            if maximize:
                sp = position[i, -1] / global_best_value if global_best_value != 0 else 1
            else:
                sp = global_best_value / position[i, -1] if position[i, -1] != 0 else 1

            if sp > p:
                x = pollination_local(position, best_global, i, nb_flower_1, nb_flower_2, min_values, max_values, r, target_function)
            else:
                x = pollination_global(position, best_global, i, gama, lamb, min_values, max_values, target_function)

            if (maximize and x[-1] >= position[i, -1]) or (not maximize and x[-1] <= position[i, -1]):
                position[i, :] = x

            current_best = position[position[:, -1].argsort()]
            value = np.copy(current_best[-1, :] if maximize else current_best[0, :])

            if (maximize and value[-1] >= best_global[-1]) or (not maximize and value[-1] <= best_global[-1]):
                best_global = np.copy(value)

        global_best_value = best_global[-1]
        count += 1

    print("\nFinal Population:")
    print(position)

    print("\nBest Solution Parameters:")
    print(best_global[:-1])

    print(f"\n{'Maximum' if maximize else 'Minimum'} Objective Value:")
    print(best_global[-1])
    return best_global

flowers = int(input("Enter the number of flowers: "))
min_values = list(map(int, input("Enter the minimum values separated by space: ").split()))
max_values = list(map(int, input("Enter the maximum values separated by space: ").split()))
iterations = int(input("Enter the number of iterations: "))
gama = float(input("Enter the scaling factor: "))
lamb = float(input("Enter the step size: "))
p = float(input("Enter the switch probability: "))
r = float(input("Enter the uniform distribution value: "))
maximize = int(input("Enter 1 for maximization or 0 for minimization: ")) == 1

fpa = flower_pollination_algorithm(
    flowers=flowers,
    min_values=min_values,
    max_values=max_values,
    iterations=iterations,
    gama=gama,
    lamb=lamb,
    p=p,
    r=r,
    target_function=custom_objective,
    maximize=maximize
)


'''OUTPUT
Enter the number of flowers: 5
Enter the minimum values separated by space: 2 5 6 
Enter the maximum values separated by space: 10 20 30
Enter the number of iterations: 100
Enter the scaling factor: 0.5
Enter the step size: 0.9
Enter the switch probability: 0.7
Enter the uniform distribution value: 0.2
Enter 1 for maximization or 0 for minimization: 1

Final Population:
[[  8.72488945   9.25383287  17.20650118  87.57694818]
 [ 10.          20.          30.         150.        ]
 [  2.68674685  16.12227445  14.29966788  80.51704624]
 [ 10.          20.          30.         150.        ]
 [  3.64949509   7.86181653   9.9859312   52.98041683]]

Best Solution Parameters:
[10. 20. 30.]

Maximum Objective Value:
150.0
'''