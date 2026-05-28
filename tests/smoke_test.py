"""
Быстрая проверка без PSO и без окон (для CI / после правок).

Запуск из корня:
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from oricon.physics.attitude import to_rodrigo_hamilton, to_transition_matrix
from oricon.physics.simulation import simulate_dynamics
from oricon.optimizers.objectives import stabilization_phi
import oricon.visualization.satellite_3d as satellite_3d_module


def main():
    lambd_0 = to_rodrigo_hamilton(0.0, 0.0, 0.0)
    lambd_1 = to_rodrigo_hamilton(0.5, 0.5, 0.5)
    matrix_1 = to_transition_matrix(lambd_1)

    # Фиктивное управление (середина допустимого диапазона)
    agent = np.array([[10.0], [3.5e6], [2000.0], [2.0e9]])

    y, t = simulate_dynamics(0, 0, 0, lambd_0, matrix_1, agent, 1, 5.0, h=0.1)
    assert y.shape[0] == 7 and len(t) > 1, "неверная форма решения"

    phi_val = stabilization_phi(0, 0, 0, lambd_0, lambd_1, matrix_1, agent, 1, 5.0, 0.1)
    assert np.isfinite(phi_val), "Phi не конечна"

    assert hasattr(satellite_3d_module, "animate_satellite_stabilization")

    print("smoke_test: OK")
    print(f"  шагов интегрирования: {len(t)}, Phi = {phi_val:.6e}")


if __name__ == "__main__":
    main()
