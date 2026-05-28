# Создание 3D-анимации: Земля + спутник на орбите со стабилизацией ориентации.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from oricon.physics.attitude import from_rodrigo_hamilton, to_transition_matrix
from oricon.physics.constants import I_ANGLE, OMEGA_E, OMEGA_0, R, R_E
from oricon.visualization.satellite_3d import (
    _draw_box,
    _orientation_error_deg,
    _satellite_model_parts,
    _transform_vertices,
)


#Радиус земли 
def _orbit_radius_display():
    return R / R_E

# фунция положения КА в инерциальной СК (ECI)
def _orbit_position_eci(u, orbit_radius, inclination):
    x = orbit_radius * np.cos(u)
    y = orbit_radius * np.sin(u) * np.cos(inclination)
    z = orbit_radius * np.sin(u) * np.sin(inclination)
    return np.array([x, y, z])


def _orbital_frame_to_eci(u, inclination):
  
    r = _orbit_position_eci(u, 1.0, inclination)
    r_hat = r / np.linalg.norm(r)

    tangent = np.array(
        [
            -np.sin(u),
            np.cos(u) * np.cos(inclination),
            np.cos(u) * np.sin(inclination),
        ]
    )
    tangent = tangent / np.linalg.norm(tangent)

    h = np.cross(r_hat, tangent)
    h = h / np.linalg.norm(h)

    return np.column_stack([tangent, h, r_hat])

# Функция для анимации стабилизации КА на орбите вокруг Земли
def _earth_rotation_angle(u):
    return (OMEGA_E / OMEGA_0) * u


def _earth_sphere_mesh(nu=28, nv=18, radius=1.0):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)
    x = radius * np.cos(uu) * np.sin(vv)
    y = radius * np.sin(uu) * np.sin(vv)
    z = radius * np.cos(vv)
    return x, y, z

# Функция поворота земли 
def _rotate_mesh(x, y, z, angle_z):
    
    c, s = np.cos(angle_z), np.sin(angle_z)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr, z


def _draw_earth(ax, earth_angle=0.0, radius=1.0):
    x, y, z = _earth_sphere_mesh(radius=radius)
    x, y, z = _rotate_mesh(x, y, z, earth_angle)
    ax.plot_surface(
        x,
        y,
        z,
        color="#1f77b4",
        alpha=0.55,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    # Упрощённые «материки» — второй слой чуть крупнее для контраста
    ax.plot_wireframe(
        x,
        y,
        z,
        color="#0d3d56",
        alpha=0.12,
        linewidth=0.3,
    )


def _draw_orbit_path(ax, orbit_radius, inclination, n=200):
    phases = np.linspace(0, 2 * np.pi, n)
    pts = np.array(
        [_orbit_position_eci(p, orbit_radius, inclination) for p in phases]
    )
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        color="#7f8c8d",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="орбита",
    )


def _draw_satellite_at(
    ax,
    center,
    rotation_eci,
    parts,
    sat_scale,
    alpha_scale=1.0,
):
    """КА в точке center; rotation_eci — ориентация в ECI."""
    for part in parts:
        verts = sat_scale * part["vertices"]
        verts = _transform_vertices(verts, rotation_eci) + center
        _draw_box(
            ax,
            verts,
            facecolors=part["facecolor"],
            edgecolor=part["edgecolor"],
            alpha=part["alpha"] * alpha_scale,
        )


def animate_satellite_earth(
    lambd_trajectory,
    u_vec,
    target_lambd=None,
    interval_ms=80,
    save_path=None,
    show=True,
    *,
    show_orbit_path=True,
    show_target_ghost=True,
    show_trail=True,
    show_hud=True,
    trail_length=60,
    box_half=0.38,
    panel_length=2.2,
    sat_scale=None,
):
    
    lambd_trajectory = np.asarray(lambd_trajectory)
    u_vec = np.asarray(u_vec)
    if len(lambd_trajectory) != len(u_vec):
        raise ValueError("lambd_trajectory и u_vec должны иметь одинаковую длину")

    n_frames = len(lambd_trajectory)
    r_orbit = _orbit_radius_display()
    inclination = I_ANGLE

    model_parts = _satellite_model_parts(box_half=box_half, panel_length=panel_length)
    if sat_scale is None:
        sat_scale = 0.09

    target_R_orb = None
    if target_lambd is not None:
        target_R_orb = to_transition_matrix(target_lambd)

    positions = np.array(
        [_orbit_position_eci(u, r_orbit, inclination) for u in u_vec]
    )

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    lim = r_orbit * 1.35 + 0.3
    hud_text = fig.text(0.02, 0.96, "", fontsize=10, va="top", family="monospace")

    def _style():
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    def update(frame):
        ax.cla()
        _style()

        earth_a = _earth_rotation_angle(u_vec[frame])
        _draw_earth(ax, earth_angle=earth_a, radius=1.0)

        if show_orbit_path:
            _draw_orbit_path(ax, r_orbit, inclination)

        pos = positions[frame]
        R_orb_eci = _orbital_frame_to_eci(u_vec[frame], inclination)
        lam = lambd_trajectory[frame]
        R_body_orb = to_transition_matrix(lam)
        R_body_eci = R_orb_eci @ R_body_orb

        if show_target_ghost and target_R_orb is not None:
            R_target_eci = R_orb_eci @ target_R_orb
            _draw_satellite_at(
                ax,
                pos,
                R_target_eci,
                model_parts,
                sat_scale,
                alpha_scale=0.2,
            )

        _draw_satellite_at(
            ax,
            pos,
            R_body_eci,
            model_parts,
            sat_scale,
            alpha_scale=1.0,
        )

        if show_trail and frame > 0:
            i0 = max(0, frame - trail_length)
            seg = positions[i0 : frame + 1]
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                seg[:, 2],
                color="#e67e22",
                linewidth=2,
                alpha=0.85,
            )

        ax.scatter([0], [0], [0], color="#2ecc71", s=20, zorder=5)
        title = f"Орбита и стабилизация КА   |   u = {u_vec[frame]:.2f}"
        ax.set_title(title, fontsize=12)

        if show_hud:
            phi, psi, theta = from_rodrigo_hamilton(lam)
            lines = [
                f"кадр {frame + 1}/{n_frames}",
                f"φ={np.degrees(phi):.1f}°  ψ={np.degrees(psi):.1f}°  θ={np.degrees(theta):.1f}°",
                f"орбита: {r_orbit:.2f} R⊕",
            ]
            if target_lambd is not None:
                err = _orientation_error_deg(lam, target_lambd)
                lines.append(f"ошибка ориентации: {err:.2f}°")
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
