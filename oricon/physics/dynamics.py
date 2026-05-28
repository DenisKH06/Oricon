import numpy as np

from oricon.physics.constants import (
    Q,
    J_A,
    DELTA,
    EPSILON,
    R,
    I_ANGLE,
    OMEGA_0,
    OMEGA_E,
    B,
    K0,
    J_STAR,
    R_E,
    G_1_1,
    G_1_0,
    H_1_1,
    G_2_0,
    G_2_1,
    G_2_2,
    H_2_1,
    H_2_2,
)
from oricon.physics.attitude import to_transition_matrix


def calc_vC(u):
    vC_1 = R * (OMEGA_0 - OMEGA_E * np.cos(I_ANGLE))
    vC_2 = R * (OMEGA_E * np.sin(I_ANGLE) * np.cos(u))
    vC_3 = 0

    vC = np.array([vC_1, vC_2, vC_3])

    return vC


def calc_M_L(matrix, matrix_1, vC, kl):
    vC_x_B = np.cross(vC, B)
    t_0 = np.matmul(matrix_1.T, vC_x_B)
    p = Q * kl * t_0
    tt = np.matmul(matrix.T, vC_x_B)

    m_L = np.cross(p, tt)
    return m_L


def calc_M_L_quadrupole(matrix, matrix_1, vC, kl, b_c):
    vC_x_b_c = np.cross(vC, b_c)
    t_0 = np.matmul(matrix_1.T, vC_x_b_c)
    p = Q * kl * t_0
    tt = np.matmul(matrix.T, vC_x_b_c)

    m_L = np.cross(p, tt)
    return m_L


def calc_domega(beta_1, beta_2, beta_3, Omega_x, Omega_y, Omega_z):
    domega_1 = OMEGA_0 * (Omega_x - beta_1)
    domega_2 = OMEGA_0 * (Omega_y - beta_2)
    domega_3 = OMEGA_0 * (Omega_z - beta_3)

    domega = np.array([domega_1, domega_2, domega_3])

    return domega


def calc_domega_vec(beta_1, beta_2, beta_3, Omega_x, Omega_y, Omega_z):
    domega_1 = OMEGA_0 * (Omega_x - beta_1)
    domega_2 = OMEGA_0 * (Omega_y - beta_2)
    domega_3 = OMEGA_0 * (Omega_z - beta_3)

    return domega_1, domega_2, domega_3


def calc_M_M(matrix, matrix_1, km):
    I_vec = km * np.matmul(matrix_1.T, B)

    m_m = np.cross(I_vec, np.matmul(matrix.T, B))

    return m_m


def calc_M_M_quadrupole(matrix, matrix_1, km, b_c):
    I_vec = km * np.matmul(matrix_1.T, b_c)

    m_m = np.cross(I_vec, np.matmul(matrix.T, b_c))

    return m_m


def calc_M_DL(matrix, vC, domega, hl):
    tt = np.matmul(matrix.T, np.cross(vC, B))

    m_dl = Q * hl * (np.cross(np.cross(domega, tt), tt))
    return m_dl


def calc_M_DL_quadrupole(matrix, vC, domega, hl, b_c):
    tt = np.matmul(matrix.T, np.cross(vC, b_c))

    m_dl = Q * hl * (np.cross(np.cross(domega, tt), tt))
    return m_dl


def calc_M_DM(matrix, domega, hm):
    m_dm = hm * (
        np.cross(np.cross(domega, np.matmul(matrix.T, B)), np.matmul(matrix.T, B))
    )

    return m_dm


def calc_M_DM_quadrupole(matrix, domega, hm, b_c):
    m_dm = hm * (
        np.cross(np.cross(domega, np.matmul(matrix.T, b_c)), np.matmul(matrix.T, b_c))
    )

    return m_dm


def calc_M_DL_modified(matrix, vC, domega, hl):
    tt = np.matmul(matrix.T, np.cross(vC, B))

    m_dl = Q * hl * np.matmul(J_STAR, (np.cross(np.cross(domega, tt), tt)))
    return m_dl


def calc_M_DL_modified_quadrupole(matrix, vC, domega, hl, b_c):
    tt = np.matmul(matrix.T, np.cross(vC, b_c))

    m_dl = Q * hl * np.matmul(J_STAR, (np.cross(np.cross(domega, tt), tt)))
    return m_dl


def calc_M_DM_modified(matrix, domega, hm):
    m_dm = hm * np.matmul(
        J_STAR,
        (np.cross(np.cross(domega, np.matmul(matrix.T, B)), np.matmul(matrix.T, B))),
    )

    return m_dm


def calc_M_DM_modified_quadrupole(matrix, domega, hm, b_c):
    m_dm = hm * np.matmul(
        J_STAR,
        (
            np.cross(
                np.cross(domega, np.matmul(matrix.T, b_c)), np.matmul(matrix.T, b_c)
            )
        ),
    )

    return m_dm


def calc_B_quadrupole(u):
    var_phi = (OMEGA_E * u) / OMEGA_0

    m1 = G_1_1
    m2 = H_1_1
    m3 = G_1_0
    m11 = 0.5 * np.sqrt(3) * G_2_2 - 0.5 * G_2_0
    m12 = 0.5 * np.sqrt(3) * H_2_2
    m13 = 0.5 * np.sqrt(3) * G_2_1
    m22 = -0.5 * G_2_0 - 0.5 * np.sqrt(3) * G_2_2
    m23 = 0.5 * np.sqrt(3) * H_2_1
    m33 = G_2_0

    b_c_xi = (-1 / R**3) * (
        R_E**3
        * (
            -(np.cos(var_phi) * m1 - np.sin(var_phi) * m2) * np.sin(u)
            + (
                np.cos(I_ANGLE) * np.sin(var_phi) * m1
                + np.cos(I_ANGLE) * np.cos(var_phi) * m2
                + np.sin(I_ANGLE) * m3
            )
            * np.cos(u)
        )
    ) - (1 / R**4) * (
        2
        * R_E**4
        * (
            -(
                np.cos(var_phi) ** 2 * m11
                - 2 * np.cos(var_phi) * np.sin(var_phi) * m12
                + np.sin(var_phi) ** 2 * m22
            )
            * np.sin(u)
            * np.cos(u)
            - (
                np.cos(var_phi) * np.cos(I_ANGLE) * np.sin(var_phi) * m11
                + np.cos(var_phi) ** 2 * np.cos(I_ANGLE) * m12
                + np.cos(var_phi) * np.sin(I_ANGLE) * m13
                - np.sin(var_phi) ** 2 * np.cos(I_ANGLE) * m12
                - np.sin(var_phi) * np.cos(I_ANGLE) * np.cos(var_phi) * m22
                - np.sin(var_phi) * np.sin(I_ANGLE) * m23
            )
            * np.sin(u) ** 2
            + (
                np.cos(var_phi) * np.cos(I_ANGLE) * np.sin(var_phi) * m11
                + np.cos(var_phi) ** 2 * np.cos(I_ANGLE) * m12
                + np.cos(var_phi) * np.sin(I_ANGLE) * m13
                - np.sin(var_phi) ** 2 * np.cos(I_ANGLE) * m12
                - np.sin(var_phi) * np.cos(I_ANGLE) * np.cos(var_phi) * m22
                - np.sin(var_phi) * np.sin(I_ANGLE) * m23
            )
            * np.cos(u) ** 2
            + (
                np.cos(I_ANGLE) ** 2 * np.sin(var_phi) ** 2 * m11
                + 2 * np.cos(I_ANGLE) ** 2 * np.sin(var_phi) * np.cos(var_phi) * m12
                + 2 * np.cos(I_ANGLE) * np.sin(var_phi) * np.sin(I_ANGLE) * m13
                + np.cos(I_ANGLE) ** 2 * np.cos(var_phi) ** 2 * m22
                + 2 * np.cos(I_ANGLE) * np.cos(var_phi) * np.sin(I_ANGLE) * m23
                + np.sin(I_ANGLE) ** 2 * m33
            )
            * np.sin(u)
            * np.cos(u)
        )
    )

    b_c_eta = (-1 / R**3) * R_E**3 * (
        -np.sin(I_ANGLE) * np.sin(var_phi) * m1
        - np.sin(I_ANGLE) * np.cos(var_phi) * m2
        + np.cos(I_ANGLE) * m3
    ) - (1 / R**4) * (
        2
        * R_E**4
        * (
            (
                -np.cos(var_phi) * np.sin(I_ANGLE) * np.sin(var_phi) * m11
                - np.cos(var_phi) ** 2 * np.sin(I_ANGLE) * m12
                + np.cos(var_phi) * np.cos(I_ANGLE) * m13
                + np.sin(var_phi) ** 2 * np.sin(I_ANGLE) * m12
                + np.sin(var_phi) * np.sin(I_ANGLE) * np.cos(var_phi) * m22
                - np.sin(var_phi) * np.cos(I_ANGLE) * m23
            )
            * np.cos(u)
            + (
                -np.cos(I_ANGLE) * np.sin(var_phi) ** 2 * np.sin(I_ANGLE) * m11
                - 2
                * np.cos(I_ANGLE)
                * np.sin(var_phi)
                * np.sin(I_ANGLE)
                * np.cos(var_phi)
                * m12
                + np.cos(I_ANGLE) ** 2 * np.sin(var_phi) * m13
                - np.cos(I_ANGLE) * np.cos(var_phi) ** 2 * np.sin(I_ANGLE) * m22
                + np.cos(I_ANGLE) ** 2 * np.cos(var_phi) * m23
                - np.sin(I_ANGLE) ** 2 * np.sin(var_phi) * m13
                - np.sin(I_ANGLE) ** 2 * np.cos(var_phi) * m23
                + np.sin(I_ANGLE) * np.cos(I_ANGLE) * m33
            )
            * np.sin(u)
        )
    )

    b_c_zeta = (1 / R**3) * (
        2
        * R_E**3
        * (
            (np.cos(var_phi) * m1 - np.sin(var_phi) * m2) * np.cos(u)
            + (
                np.cos(I_ANGLE) * np.sin(var_phi) * m1
                + np.cos(I_ANGLE) * np.cos(var_phi) * m2
                + np.sin(I_ANGLE) * m3
            )
            * np.sin(u)
        )
    ) + (1 / R**4) * (
        3
        * R_E**4
        * (
            (
                np.cos(var_phi) ** 2 * m11
                - 2 * np.cos(var_phi) * np.sin(var_phi) * m12
                + np.sin(var_phi) ** 2 * m22
            )
            * np.cos(u) ** 2
            + 2
            * (
                np.cos(var_phi) * np.cos(I_ANGLE) * np.sin(var_phi) * m11
                + np.cos(var_phi) ** 2 * np.cos(I_ANGLE) * m12
                + np.cos(var_phi) * np.sin(I_ANGLE) * m13
                - np.sin(var_phi) ** 2 * np.cos(I_ANGLE) * m12
                - np.sin(var_phi) * np.cos(I_ANGLE) * np.cos(var_phi) * m22
                - np.sin(var_phi) * np.sin(I_ANGLE) * m23
            )
            * np.sin(u)
            * np.cos(u)
            + (
                np.cos(I_ANGLE) ** 2 * np.sin(var_phi) ** 2 * m11
                + 2 * np.cos(I_ANGLE) ** 2 * np.sin(var_phi) * np.cos(var_phi) * m12
                + 2 * np.cos(I_ANGLE) * np.sin(var_phi) * np.sin(I_ANGLE) * m13
                + np.cos(I_ANGLE) ** 2 * np.cos(var_phi) ** 2 * m22
                + 2 * np.cos(I_ANGLE) * np.cos(var_phi) * np.sin(I_ANGLE) * m23
                + np.sin(I_ANGLE) ** 2 * m33
            )
            * np.sin(u) ** 2
        )
    )

    b_c = np.array([b_c_xi, b_c_eta, b_c_zeta])

    return b_c