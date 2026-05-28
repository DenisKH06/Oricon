# Целевые функции

import numpy as np

from oricon.physics.constants import OMEGA_0
from oricon.physics.simulation import simulate_dynamics
from oricon.solvers.quadrature import integrate_simpson


def stabilization_phi(
    omega_x0,
    omega_y0,
    omega_z0,
    lambd_vec_0,
    lambd_vec_target,
    matrix_1,
    agent,
    n,
    u_max,
    h,
):
    
    #Функционал Φ — интеграл квадрата ошибки ориентации и угловых скоростей.
    y, _u = simulate_dynamics(
        omega_x0,
        omega_y0,
        omega_z0,
        lambd_vec_0,
        matrix_1,
        agent,
        n,
        u_max,
        h,
    )

    lambd_vec = y[0:4, :].T
    omega_vec = y[4:7, :].T

    omega_x = OMEGA_0 * (
        omega_vec[:, 0]
        - 2 * (lambd_vec[:, 1] * lambd_vec[:, 2] + lambd_vec[:, 0] * lambd_vec[:, 3])
    )
    omega_y = OMEGA_0 * (
        omega_vec[:, 1]
        - 2
        * (
            lambd_vec[:, 0] ** 2
            - lambd_vec[:, 1] ** 2
            + lambd_vec[:, 2] ** 2
            - lambd_vec[:, 3] ** 2
        )
    )
    omega_z = OMEGA_0 * (
        omega_vec[:, 2]
        - 2 * (lambd_vec[:, 2] * lambd_vec[:, 3] - lambd_vec[:, 0] * lambd_vec[:, 1])
    )

    total_error = (
        np.sum((lambd_vec - lambd_vec_target) ** 2, axis=1)
        + omega_x * omega_x
        + omega_y * omega_y
        + omega_z * omega_z
    )

    return (h / 2) * np.sum(total_error[:-1] + total_error[1:])


