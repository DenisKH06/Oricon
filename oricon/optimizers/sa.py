import numpy as np
from tqdm import tqdm
import math

from oricon.optimizers.objectives import stabilization_phi  


def run_sa(
    n,
    max_iter,
    h,
    u_max,
    Omega_x0,
    Omega_y0,
    Omega_z0,
    param_min,
    param_max,
    lambd_vec_0,
    lambd_vec_1,
    matrix_1,
    t_max=1000.0,  # начальная температура
    t_min=1e-9,  # температура остановки
    e_min=1e-6,  # энергия остановки
    alpha=0.98,  # коэффициент уменьшения температуры
    candidat_coef=0.1,  # для поиска кандидатов
):
    # Инициализируем температуру
    t = t_max

    # Начальное решение
    current_solution = np.zeros((4, n))
    for i in range(4):
        current_solution[i] = np.random.uniform(param_min[i], param_max[i], n)

    # Значение целевой функции для начального решения (для SA называется энергией)
    current_energy = stabilization_phi(
        Omega_x0,
        Omega_y0,
        Omega_z0,
        lambd_vec_0,
        lambd_vec_1,
        matrix_1,
        current_solution,
        n,
        u_max,
        h,
    )

    # Лучшее решение
    best_solution = current_solution.copy()
    best_energy = current_energy

    # === SA
    for _ in tqdm(range(max_iter), desc="Итерации SA", unit="итерация"):
        # Проверка условий остановки
        if t < t_min or current_energy < e_min:
            break

        # 2.1. Генерация нового кандидатного решения (возмущение)
        new_solution = candidate(current_solution, param_min, param_max, candidat_coef)

        # Вычисляем целевую функцию (энергию для SA)
        new_energy = stabilization_phi(
            Omega_x0,
            Omega_y0,
            Omega_z0,
            lambd_vec_0,
            lambd_vec_1,
            matrix_1,
            new_solution,
            n,
            u_max,
            h,
        )

        # Находим разницу энергий (нового решения и текущего)
        delta_e = new_energy - current_energy

        # Решаем, принимаем новое решение или нет
        if accept_or_not(delta_e, t):
            # Принимаем
            current_solution = new_solution.copy()
            current_energy = new_energy

            # Обновляем лучшее решение, если было какое-то улучшение
            if new_energy < best_energy:
                best_solution = new_solution.copy()
                best_energy = new_energy

        # Понижаем температуру
        t = alpha * t

    return best_solution, best_energy


def candidate(current_solution, param_min, param_max, candidat_coef):
    new_solution = current_solution.copy()

    for i in range(4):
        # Допустимый диапазон
        param_range = param_max[i] - param_min[i]

        # Для смещения
        m = np.random.uniform(
            -candidat_coef * param_range,
            candidat_coef * param_range,
            current_solution[i].shape,
        )

        # Применяем возмущение
        new_solution[i] += m

        # Обеспечиваем соблюдение границ параметров
        new_solution[i] = np.clip(new_solution[i], param_min[i], param_max[i])

    return new_solution


def accept_or_not(delta_e, t):
    if delta_e < 0:
        # Всегда принимаем решения с лучшей энергией
        return True
    else:
        # С вероятностью exp(-delta_e/T) принимаем решение с худшей энергией
        probability = math.exp(-delta_e / t)
        return np.random.random() < probability