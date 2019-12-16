from math import sin, pi, sqrt, pow
from random import uniform
import numpy as np


def function(x):
    return sin(x) + 0.5


def noise(func_value, a):
    noise_value = uniform(-a, a)
    func_value += noise_value


def mean_harmonic(func_values, weights):
    

def noisiness_criterion(func_values):
    noise_values = np.zeros(len(func_values))

    for i in range(1, len(func_values)):
        noise_values[i] = pow(func_values[i] - func_values[i - 1], 2)

    return sqrt(noise_values.sum())


def difference_criterion(func_values, filtered_func_values):
    diff_values = np.zeros(len(func_values))

    for i in range(len(func_values)):
        diff_values[i] = pow(filtered_func_values[i] - func_values[i], 2)

    return sqrt(diff_values.sum() / len(diff_values))


def distance(noisiness, difference):
    return sqrt(pow(noisiness, 2) + pow(difference, 2))


class Filter:
    def __init__(self):
        self.func_values = np.array([])

        # Parameters
        self.noise_amplitude = 0.25
        self.prob = 0.95
        self.epsilon = 0.01
        self.window_size_1 = 3
        self.window_size_2 = 5

        self.function = function