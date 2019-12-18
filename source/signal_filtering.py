from math import sin, pi, sqrt, pow, fabs, log, ceil, floor
from random import uniform
import numpy as np
import matplotlib.pyplot as plt


# Task function, that simulates a signal
def function(x):
    return sin(x) + 0.5


# Noise superimposing function
def noise(func_values, a):
    noised_func_values = func_values.copy()
    noise_value = 0

    for i in range(len(noised_func_values)):
        noise_value = uniform(-a, a)
        noised_func_values[i] += noise_value

    return noised_func_values


# Function for calculating mean harmonic value of window sector
def mean_harmonic(func_values, weights):
    curr_sum = 0
    for i in range(len(func_values)):
        curr_sum += weights[i] / func_values[i]
    return 1 / curr_sum


# Function for calculating the noisiness criterion of function
def calc_noisiness(func_values):
    noise_values = np.zeros(len(func_values))

    for i in range(1, len(func_values)):
        noise_values[i] = pow(func_values[i] - func_values[i - 1], 2)

    return sqrt(noise_values.sum())


# Function for calculating the difference criterion for two functions
def calc_difference(func_values, filtered_func_values):
    diff_values = np.zeros(len(func_values))

    for i in range(len(func_values)):
        diff_values[i] = pow(filtered_func_values[i] - func_values[i], 2)

    return sqrt(diff_values.sum() / len(diff_values))


# Function for calculating Euclidean distance between two points
def distance(noisiness, difference):
    return sqrt(pow(noisiness, 2) + pow(difference, 2))


# Function for calculating number of experiments in random search, needed for achieving 95% probability of success
def calc_number_of_tries(probability, epsilon, x_max, x_min):
    return ceil(fabs(log(1 - probability) / log(1 - (epsilon / (x_max - x_min)))))


# Function that generates set of weights (alphas)
def generate_weights(window_size):
    if window_size == 3:
        weights = np.zeros(3)
        weights[1] = uniform(0, 1)  # Middle
        weights[0] = (1 - weights[1]) / 2  # Left
        weights[2] = weights[0]  # Right
        return weights
    else:
        weights = np.zeros(5)
        weights[2] = uniform(0, 1)  # Middle
        weights[1] = uniform(0, 1 - weights[2]) / 2  # Middle-left
        weights[3] = weights[1]  # Middle-right
        weights[0] = (1 - weights.sum()) / 2  # Left
        weights[4] = weights[0]  # Right
        return weights


def print_table(lambdas, experiments, window_size):
    print()
    print("Results of experiments for r=" + str(window_size) + ":")
    print(" _______________________________________________________________________________________ ")
    print("|  λ  |    J   | Distance |              Weights               | Noisiness | Difference |")
    print("|-----|--------|----------|------------------------------------|-----------|------------|")
    for i in range(len(lambdas)):
        print("| ", end="")
        print(lambdas[i], end=' | ')
        print('{0:.4f}'.format(experiments[i][0]), end=' |  ')
        print('{0:.4f}'.format(distance(experiments[i][2], experiments[i][3])), end='  | ')

        if window_size == 3:
            print("      ", end='')

        for k in range(window_size):
            print('{0:.4f}'.format(experiments[i][1][k]), end='  ')

        if window_size == 3:
            print("     ", end='')

        print("| ", end="  ")
        print('{0:.4f}'.format(experiments[i][2]), end="  |   ")
        print('{0:.4f}'.format(experiments[i][3]), end="   |\n")

    print("|_____|________|__________|____________________________________|___________|____________|")
    print()
    print()

    print("Best values:")
    best_index = experiments.index(min(experiments, key=lambda x: distance(x[2], x[3])))  # Compare by distance
    print(" _______________________________________________________________________________________ ")
    print("|  λ  |    J   | Distance |              Weights               | Noisiness | Difference |")
    print("|-----|--------|----------|------------------------------------|-----------|------------|")

    print("| ", end="")
    print(lambdas[best_index], end=' | ')
    print('{0:.4f}'.format(experiments[best_index][0]), end=' |  ')
    print('{0:.4f}'.format(distance(experiments[best_index][2], experiments[best_index][3])), end='  | ')

    if window_size == 3:
        print("       ", end='')
    for k in range(window_size):
        print('{0:.4f}'.format(experiments[best_index][1][k]), end=' ')
    if window_size == 3:
        print("       ", end='')

    print("| ", end="  ")
    print('{0:.4f}'.format(experiments[best_index][2]), end="  |   ")
    print('{0:.4f}'.format(experiments[best_index][3]), end="   |\n")

    print("|_____|________|__________|____________________________________|___________|____________|")


def print_J_points(experiments):
    plt.scatter(0, 0)
    for i in range(len(experiments)):
        plt.scatter(experiments[i][2], experiments[i][3])

    # Decoration
    plt.title('J points')
    plt.ylabel('difference')
    plt.xlabel('noisiness')

    legend = ["utopia point"]
    for i in range(len(experiments)):
        legend.append(str(i / 10))

    plt.legend(legend, loc='upper left')

    plt.show()


class Filter:
    def __init__(self, window_size):
        self.func_values = []
        self.noised_func_values = []
        self.filtered_func_values = []

        # Parameters
        self.noise_amplitude = 0.25
        self.probability = 0.95
        self.epsilon = 0.01
        self.window_size = window_size
        self.M = int((window_size - 1) / 2)
        self.number_of_samples = 100
        self.L = 10
        self.x_max = pi
        self.x_min = 0
        self.a = 0.25

        # Results initialization
        self.best_lambda = 0
        self.best_J = 0
        self.best_weights = np.zeros(window_size)
        self.best_noisiness = 0
        self.best_difference = 0

        # Function values initializing
        for k in range(self.number_of_samples):
            self.func_values.append(function(self.x_min + k * (self.x_max - self.x_min) / self.number_of_samples))
        self.noised_func_values = noise(self.func_values, self.a)

    def filter(self):
        # Calculate parameters
        res_lambda, res_parameters = self.find_best_parameters()

        # Summarize
        self.best_lambda = res_lambda
        self.best_J = res_parameters[0]
        self.best_weights = res_parameters[1]
        self.best_noisiness = res_parameters[2]
        self.best_difference = res_parameters[3]

        # Filter signal
        window_elements = []  # Elements in current window sector
        for i in range(self.M, len(self.noised_func_values) - self.M):
            for k in range(self.window_size):
                window_elements.append(self.noised_func_values[i - self.M + k])
            self.filtered_func_values.append(mean_harmonic(window_elements, self.best_weights))
            window_elements.clear()

    def filter_with_weights(self, weights):
        curr_filtered_func_values = []  # New func values after
        window_elements = []  # Elements in current window sector

        for i in range(self.M, len(self.noised_func_values) - self.M):
            for k in range(self.window_size):
                window_elements.append(self.noised_func_values[i - self.M + k])
            curr_filtered_func_values.append(mean_harmonic(window_elements, weights))
            window_elements.clear()

        return curr_filtered_func_values

    def find_best_parameters(self):
        lambdas = []
        experiments = []

        for i in range(self.L + 1):
            lambdas.append(i / self.L)

        for curr_lambda in lambdas:
            experiments.append(self.find_best_J(curr_lambda))

        best_index = experiments.index(min(experiments, key=lambda x: distance(x[2], x[3])))  # Compare by distance

        print_table(lambdas, experiments, self.window_size)
        print_J_points(experiments)

        return lambdas[best_index], experiments[best_index]

    def find_best_J(self, curr_lambda):
        weights_hist = []
        noisiness_hist = []
        difference_hist = []
        J_values_hist = []

        for i in range(calc_number_of_tries(self.probability, self.epsilon, self.x_max, self.x_min)):
            curr_weights = generate_weights(self.window_size)
            curr_func_values = self.filter_with_weights(curr_weights)
            curr_noisiness = calc_noisiness(curr_func_values)
            curr_difference = calc_difference(curr_func_values, self.noised_func_values)

            J_value = curr_lambda * curr_noisiness + (1 - curr_lambda) * curr_difference

            # Append to history
            weights_hist.append(curr_weights)
            noisiness_hist.append(curr_noisiness)
            difference_hist.append(curr_difference)
            J_values_hist.append(J_value)

        min_index = J_values_hist.index(min(J_values_hist))

        # return result in format: J, weights, noisiness, difference
        return J_values_hist[min_index], weights_hist[min_index], noisiness_hist[min_index], difference_hist[min_index]

    def visualize_filtering(self):
        x = []
        for k in range(self.number_of_samples):
            x.append(self.x_min + k * (self.x_max - self.x_min) / self.number_of_samples)

        short_x = []
        for k in range(self.M, self.number_of_samples - self.M):
            short_x.append(self.x_min + k * (self.x_max - self.x_min) / self.number_of_samples)

        plt.plot(x, self.func_values)
        plt.plot(x, self.noised_func_values)
        plt.plot(short_x, self.filtered_func_values)

        # Decoration
        plt.title('Signal')
        plt.ylabel('f(x)')
        plt.xlabel('x')
        plt.legend(['f(x) = sin(x) + 0.5', 'noise', 'filtering with r=' + str(self.window_size)], loc='lower center')

        plt.show()
