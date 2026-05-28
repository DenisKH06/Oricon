import numpy as np
from tqdm import tqdm

from oricon.optimizers.objectives import stabilization_phi
from oricon.optimizers.pso_coefficients import c1_up, c2_up, get_coefficient_method


def run_pso(
    s,
    n,
    max_iter,
    h,
    u_max,
    omega_x0,
    omega_y0,
    omega_z0,
    param_min,
    param_max,
    lambd_vec_0,
    lambd_vec_target,
    matrix_1,
    method_name="linear"
):
    method = get_coefficient_method(method_name)

    particles = np.zeros((s, 4, n))
    velocities = np.zeros((s, 4, n))

    for i in range(4):
        particles[:, i, :] = np.random.uniform(param_min[i], param_max[i], (s, n))
        velocities[:, i, :] = np.random.uniform(
            0.15 * param_min[i], 0.15 * param_max[i], (s, n)
        )

    pbest = particles.copy()
    value_at_pbest = np.full(s, np.inf)
    gbest = particles[0].copy()
    value_at_gbest = np.inf
    convergence = []

    for iteration in tqdm(range(max_iter), desc="Итерации", unit="итерация"):
        for j in range(s):
            current = stabilization_phi(
                omega_x0,
                omega_y0,
                omega_z0,
                lambd_vec_0,
                lambd_vec_target,
                matrix_1,
                particles[j],
                n,
                u_max,
                h,
            )
            if current < value_at_pbest[j]:
                value_at_pbest[j] = current
                pbest[j] = particles[j].copy()
            if current < value_at_gbest:
                value_at_gbest = current
                gbest = particles[j].copy()

        convergence.append(value_at_gbest)
        w, c1, c2 = method(iteration, max_iter)

        r1 = np.random.uniform(0, c1_up, (s, 4, n))
        r2 = np.random.uniform(0, c2_up, (s, 4, n))
        velocities = (
            w * velocities + c1 * r1 * (gbest - particles) + c2 * r2 * (pbest - particles)
        )
        particles += velocities

        for k in range(4):
            mask_min = particles[:, k, :] < param_min[k]
            mask_max = particles[:, k, :] > param_max[k]
            particles[:, k, :][mask_min] = np.random.uniform(
                param_min[k], param_max[k], mask_min.sum()
            )
            particles[:, k, :][mask_max] = np.random.uniform(
                param_min[k], param_max[k], mask_max.sum()
            )

            mask_min = velocities[:, k, :] < 0.15 * param_min[k]
            mask_max = velocities[:, k, :] > 0.15 * param_max[k]
            velocities[:, k, :][mask_min] = np.random.uniform(
                0.15 * param_min[k], 0.15 * param_max[k], mask_min.sum()
            )
            velocities[:, k, :][mask_max] = np.random.uniform(
                0.15 * param_min[k], 0.15 * param_max[k], mask_max.sum()
            )

    return gbest, value_at_gbest, convergence
