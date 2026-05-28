import numpy as np

"параметры для разных методов подсчёта коэффициентов в PSO"

w_low, c1_low, c2_low = 0.4, 0.0, 0.0
w_up, c1_up, c2_up = 0.9, 2.05, 2.05

w_low_random, c1_low_random, c2_low_random = 0, 0, 0
w_max_random, c1_max_random, c2_max_random = 1, 1, 1

w_constant, c1_constant, c2_constant = 0.7, 1.5, 1.5


def constants_coef(iter, max_iter):
    return w_constant, c1_constant, c2_constant


def linear_coef(iter, max_iter):
    w = (w_low - w_up) * (iter / max_iter) + w_up
    c1 = (c1_low - c1_up) * (iter / max_iter) + c1_up
    c2 = (c2_up - c2_low) * (iter / max_iter) + c2_low
    return w, c1, c2


def random_coef(iter, max_iter):
    w = np.random.uniform(w_low_random, w_max_random)
    c1 = np.random.uniform(c1_low_random, c1_max_random)
    c2 = np.random.uniform(c2_low_random, c2_max_random)
    return w, c1, c2


def combined_coef(iter, max_iter):
    w = (w_low - w_up) * (iter / max_iter) + w_up
    c1 = (c1_low - c1_up) * (iter / max_iter) + c1_up
    c2 = (c2_up - c2_low) * (iter / max_iter) + c2_low
    w *= np.random.uniform(w_low_random, w_max_random)
    c1 *= np.random.uniform(c1_low_random, c1_max_random)
    c2 *= np.random.uniform(c2_low_random, c2_max_random)
    return w, c1, c2


coef_methods = {
    "constants": constants_coef,
    "linear": linear_coef,
    "random": random_coef,
    "combined": combined_coef,
}


def get_coefficient_method(method_name):
    return coef_methods.get(method_name, linear_coef)
