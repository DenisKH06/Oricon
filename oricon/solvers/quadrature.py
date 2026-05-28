#Другие численные интегриторы, например, метод Симпсона.

import numpy as np
from scipy.integrate import quad

def integrate_simpson(y: np.ndarray, h: float) -> float:
    return (h / 2) * np.sum(y[:-1] + y[1:])
