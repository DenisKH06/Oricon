import matplotlib.pyplot as plt
import numpy as np


def plot_angles(phi_values, psi_values, theta_values, u_vec):
    plt.figure(figsize=(12, 5))
    plt.plot(u_vec, phi_values, label="φ (phi)", color="blue")
    plt.plot(u_vec, psi_values, label="ψ (psi)", color="green")
    plt.plot(u_vec, theta_values, label="θ (theta)", color="red")
    plt.xlabel("u")
    plt.ylabel("рад.")
    plt.title("Изменение углов φ, ψ, θ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_domega(domega_1, domega_2, domega_3, u_vec):
    plt.figure(figsize=(12, 5))
    plt.plot(u_vec, domega_1, label="dω1", color="blue")
    plt.plot(u_vec, domega_2, label="dω2", color="green")
    plt.plot(u_vec, domega_3, label="dω3", color="red")
    plt.xlabel("u")
    plt.ylabel("Проекции domega")
    plt.title("Изменение проекций domega")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lambd(lambd_vec, u_vec):
    plt.figure(figsize=(12, 5))
    for i, color in enumerate(["blue", "green", "red", "black"]):
        plt.plot(u_vec, lambd_vec[:, i], label=f"λ{i}", color=color)
    plt.xlabel("u")
    plt.ylabel("λ")
    plt.title("Параметры Родрига–Гамильтона")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_convergence(convergence):
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(convergence)), convergence, label="Φ", color="blue")
    plt.xlabel("Итерация")
    plt.ylabel("Значение целевой функции")
    plt.title("Сходимость оптимизации")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kl(kl, n, u_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(x, np.append(kl, kl[-1]), where="post", label="kl", color="blue")
    plt.xlabel("u")
    plt.ylabel("kl")
    plt.title("Изменение параметра kl")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_km(km, n, u_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(x, np.append(km, km[-1]), where="post", label="km", color="black")
    plt.xlabel("u")
    plt.ylabel("km")
    plt.title("Изменение параметра km")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hl(hl, n, u_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(x, np.append(hl, hl[-1]), where="post", label="hl", color="green")
    plt.xlabel("u")
    plt.ylabel("hl")
    plt.title("Изменение параметра hl")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hm(hm, n, u_max):
    plt.figure(figsize=(12, 5))
    x = np.linspace(0, u_max, n + 1)
    plt.step(x, np.append(hm, hm[-1]), where="post", label="hm", color="red")
    plt.xlabel("u")
    plt.ylabel("hm")
    plt.title("Изменение параметра hm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


