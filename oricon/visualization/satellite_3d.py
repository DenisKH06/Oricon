#3D-анимация стабилизации спутника по траектории параметров Родрига–Гамильтона.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from oricon.physics.attitude import from_rodrigo_hamilton, to_transition_matrix

# Функция, которая генерирует массив вершин
def _box_vertices(x0, x1, y0, y1, z0, z1):
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )


def _satellite_model_parts(
    box_half=0.38,
    panel_length=2.2,
    panel_width=0.36,
    panel_thick=0.04,
):
    
    a = box_half
    w = panel_width
    t = panel_thick
    L = panel_length

    parts = [
        {
            "name": "box",
            "vertices": _box_vertices(-a, a, -a, a, -a, a),
            "facecolor": "#bdc3c7",
            "edgecolor": "#2c3e50",
            "alpha": 0.92,
        },
        {
            "name": "panel_pos_y",
            "vertices": _box_vertices(-w, w, a, a + L, -t, t),
            "facecolor": "#1a5276",
            "edgecolor": "#154360",
            "alpha": 0.88,
        },
        {
            "name": "panel_neg_y",
            "vertices": _box_vertices(-w, w, -(a + L), -a, -t, t),
            "facecolor": "#1a5276",
            "edgecolor": "#154360",
            "alpha": 0.88,
        },
        # Тонкие «балки» между корпусом и панелями
        {
            "name": "stick_pos",
            "vertices": _box_vertices(-0.06, 0.06, a, a + 0.12, -0.06, 0.06),
            "facecolor": "#566573",
            "edgecolor": "#1c2833",
            "alpha": 0.9,
        },
        {
            "name": "stick_neg",
            "vertices": _box_vertices(-0.06, 0.06, -(a + 0.12), -a, -0.06, 0.06),
            "facecolor": "#566573",
            "edgecolor": "#1c2833",
            "alpha": 0.9,
        },
    ]
    return parts


# Функция отрисовки всех частей КА
def _draw_satellite_model(ax, rotation_matrix, parts, alpha_scale=1.0):

    for part in parts:
        verts = _transform_vertices(part["vertices"], rotation_matrix)
        _draw_box(
            ax,
            verts,
            facecolors=part["facecolor"],
            edgecolor=part["edgecolor"],
            alpha=part["alpha"] * alpha_scale,
        )

# Считаем размер модели по координатам (Грубая оценка модели КА)
def _model_extent(parts):
    all_v = np.vstack([p["vertices"] for p in parts])
    return float(np.max(np.abs(all_v)))

# Получаем грани
def _box_faces():
    return [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]

# Поворачиваем вершины в соответствии с ориентацией КА
def _transform_vertices(vertices, rotation_matrix):
    return (rotation_matrix @ vertices.T).T

# Функция отрисовки неподвижных осей орбитальной СК
def _draw_orbit_axes(ax, length=1.8):
    
    origin = np.zeros(3)
    labels = ("ξ", "η", "ζ")
    colors = ["#c44", "#4a4", "#44c"]
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        e = np.zeros(3)
        e[i] = length
        ax.quiver(
            *origin,
            *e,
            color=color,
            linewidth=1.2,
            linestyle="dashed",
            alpha=0.45,
            arrow_length_ratio=0.12,
        )
        ax.text(*(e * 1.15), label, color=color, fontsize=8, alpha=0.7)

# Функция отрисовки осей, связанных с корпусом КА
def _draw_body_axes(ax, rotation_matrix, length=1.4):
    
    origin = np.zeros(3)
    names = ("Xк", "Yк", "Zк")
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    
    for i, (name, color) in enumerate(zip(names, colors)):
        d = rotation_matrix[:, i] * length
        ax.quiver(
            *origin,
            *d,
            color=color,
            linewidth=2.5,
            arrow_length_ratio=0.12,
        )
        ax.text(*(d * 1.08), name, color=color, fontsize=9, fontweight="bold")


def _draw_box(ax, vertices, facecolors, edgecolor="navy", alpha=0.5):
    
    faces_idx = _box_faces()
    borders = [vertices[idx] for idx in faces_idx]
    fc = facecolors
    border = Poly3DCollection(
        borders,
        facecolors=fc,
        edgecolors=edgecolor,
        alpha=alpha,
        linewidths=0.9,
    )
    ax.add_collection3d(border)

# Функция для вычисления угла между текущей ориентацией и целевой (в градусах)
def _orientation_error_deg(lambd_current, lambd_target):
    
    r0 = to_transition_matrix(lambd_current)
    r1 = to_transition_matrix(lambd_target)
    r_rel = r0.T @ r1
    trace = np.clip((np.trace(r_rel) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def animate_satellite_stabilization(
    lambd_trajectory,
    target_lambd=None,
    u_vec=None,
    interval_ms=80,
    save_path=None,
    show=True,
    *,
    show_orbit_axes=True,
    show_body_axes=True,
    show_target_ghost=True,
    show_axis_trail=True,
    show_hud=True,
    trail_length=80,
    box_half=0.38,
    panel_length=2.2,
):
        
    lambd_trajectory = np.asarray(lambd_trajectory)
    n_frames = len(lambd_trajectory)
    model_parts = _satellite_model_parts(
        box_half=box_half,
        panel_length=panel_length,
    )
    extent = _model_extent(model_parts)

    target_matrix = None
    if target_lambd is not None and show_target_ghost:
        target_matrix = to_transition_matrix(target_lambd)


    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    lim = extent * 1.15 + 0.5
    hud_text = fig.text(0.02, 0.96, "", fontsize=10, va="top", family="monospace")

    def _style_axes():
        
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("ξ")
        ax.set_ylabel("η")
        ax.set_zlabel("ζ")
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    def update(frame):
        ax.cla()
        _style_axes()

        if show_orbit_axes:
            _draw_orbit_axes(ax)

        if target_matrix is not None:
            _draw_satellite_model(ax, target_matrix, model_parts, alpha_scale=0.14)

        lam = lambd_trajectory[frame]
        matrix = to_transition_matrix(lam)
        _draw_satellite_model(ax, matrix, model_parts, alpha_scale=1.0)

        if show_body_axes:
            _draw_body_axes(ax, matrix)

        
        title = "Стабилизация ориентации КА"
        if u_vec is not None:
            title += f"   |   u = {u_vec[frame]:.2f}"
        ax.set_title(title, fontsize=12)

        if show_hud:
            phi, psi, theta = from_rodrigo_hamilton(lam)
            lines = [
                f"кадр {frame + 1}/{n_frames}",
                f"φ={np.degrees(phi):.1f}°  ψ={np.degrees(psi):.1f}°  θ={np.degrees(theta):.1f}°",
            ]
            if target_lambd is not None:
                err = _orientation_error_deg(lam, target_lambd)
                lines.append(f"ошибка до цели: {err:.2f}°")
            hud_text.set_text("\n".join(lines))
        else:
            hud_text.set_text("")

        return []


    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_path:
        fps = max(1, 1000 // interval_ms)
        if save_path.lower().endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps)

    if show:
        plt.show()
    elif not save_path:
        plt.close(fig)

    return anim
