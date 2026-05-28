import numpy as np

from oricon.physics.attitude import to_transition_matrix
from oricon.physics.constants import DELTA, EPSILON, J_A, K0, OMEGA_0
from oricon.physics.dynamics import (
    calc_B_quadrupole,
    calc_M_DL_quadrupole,
    calc_M_DM_quadrupole,
    calc_M_L_quadrupole,
    calc_M_M_quadrupole,
    calc_domega,
    calc_vC,
)
from oricon.solvers.ode import solve_ode


def make_satellite_rhs(matrix_1, agent, n, u_max):

    interval = u_max / n

    def rhs(u, state):
        lambd_0, lambd_1, lambd_2, lambd_3, omega_x, omega_y, omega_z = state
        index = min(int(u // interval), n - 1)
        kl, km, hl, hm = (
            agent[0, index],
            agent[1, index],
            agent[2, index],
            agent[3, index],
        )

        matrix = to_transition_matrix([lambd_0, lambd_1, lambd_2, lambd_3])
        beta_1, beta_2, beta_3 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
        v_c = calc_vC(u)
        domega = calc_domega(beta_1, beta_2, beta_3, omega_x, omega_y, omega_z)
        b_c = calc_B_quadrupole(u)

        m_l = calc_M_L_quadrupole(matrix, matrix_1, v_c, kl, b_c)
        m_m = calc_M_M_quadrupole(matrix, matrix_1, km, b_c)
        m_dl = calc_M_DL_quadrupole(matrix, v_c, domega, hl, b_c)
        m_dm = calc_M_DM_quadrupole(matrix, domega, hm, b_c)

        norm_sq = lambd_0**2 + lambd_1**2 + lambd_2**2 + lambd_3**2
        dlambd_0 = 0.5 * (
            -lambd_1 * omega_x
            - lambd_2 * omega_y
            - lambd_3 * omega_z
            + lambd_2
            - K0 * lambd_0 * (norm_sq - 1)
        )
        dlambd_1 = 0.5 * (
            lambd_0 * omega_x
            + lambd_2 * omega_z
            - lambd_3 * omega_y
            - lambd_3
            - K0 * lambd_1 * (norm_sq - 1)
        )
        dlambd_2 = 0.5 * (
            lambd_0 * omega_y
            + lambd_3 * omega_x
            - lambd_1 * omega_z
            - lambd_0
            - K0 * lambd_2 * (norm_sq - 1)
        )
        dlambd_3 = 0.5 * (
            lambd_0 * omega_z
            + lambd_1 * omega_y
            - lambd_2 * omega_x
            + lambd_1
            - K0 * lambd_3 * (norm_sq - 1)
        )
        d_omega_x = -(EPSILON - DELTA) * (omega_y * omega_z - beta_2 * beta_3) + (
            m_l[0] + m_m[0] + m_dl[0] + m_dm[0]
        ) / (J_A * OMEGA_0**2)
        d_omega_y = -((1 - EPSILON) / DELTA) * (
            omega_z * omega_x - beta_1 * beta_3
        ) + (m_l[1] + m_m[1] + m_dl[1] + m_dm[1]) / (DELTA * J_A * OMEGA_0**2)
        d_omega_z = -((DELTA - 1) / EPSILON) * (
            omega_x * omega_y - beta_2 * beta_1
        ) + (m_l[2] + m_m[2] + m_dl[2] + m_dm[2]) / (EPSILON * J_A * OMEGA_0**2)

        return [
            dlambd_0,
            dlambd_1,
            dlambd_2,
            dlambd_3,
            d_omega_x,
            d_omega_y,
            d_omega_z,
        ]

    return rhs


def simulate_dynamics(
    omega_x0,
    omega_y0,
    omega_z0,
    lambd_vec_0,
    matrix_1,
    agent,
    n,
    u_max,
    h=0.05,
    method="RK45",
):
    
    times = np.arange(0, u_max + h, h)
    rhs = make_satellite_rhs(matrix_1, agent, n, u_max)
    y0 = np.concatenate([lambd_vec_0, [omega_x0, omega_y0, omega_z0]])
    result = solve_ode(rhs, (0, u_max), y0, t_eval=times, method=method)
    return result.y, result.t



