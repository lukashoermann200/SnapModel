import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def plot_debug_pes(func, apex_pos, phi, save_dir=".", show=False):
    """
    Plot and save a 1D potential energy surface (PES) for varying θ at fixed φ.

    Parameters
    ----------
    func : callable
        Function that takes [theta] as input and returns energy in eV.
    apex_pos : array-like
        Position of the apex for title annotation.
    phi : float
        Fixed phi angle in radians.
    save_dir : str
        Directory to save output files.
    show : bool
        Whether to display the plot interactively.
    """
    x_array = np.linspace(-0.5, 0.5, 50)
    U_array = np.array([func(np.array([x])) for x in x_array])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        save_dir / f"AFM_PES_{timestamp}.csv",
        np.column_stack((x_array, U_array)),
        delimiter=",",
    )

    fig, ax = plt.subplots()
    ax.set_title(f"{apex_pos} | φ = {phi * 180 / np.pi:.1f}°")
    ax.plot(x_array, U_array)
    ax.set_xlabel(r"$\theta$ / rad")
    ax.set_ylabel("Energy / eV")
    fig.tight_layout()
    fig.savefig(save_dir / f"AFM_PES_{timestamp}.png", dpi=300)

    if show:
        plt.show()
    plt.close(fig)


def plot_debug_pes_2d(func, apex_pos, theta0, phi0, save_dir=".", show=False):
    """
    Plot and save a 2D polar PES map over θ and φ.

    Parameters
    ----------
    func : callable
        Function taking [theta, phi] and returning energy in eV.
    apex_pos : array-like
        Position of the apex for title annotation.
    theta0 : float
        Reference theta value (e.g. equilibrium) in radians.
    phi0 : float
        Reference phi value (e.g. equilibrium) in radians.
    save_dir : str
        Directory to save output files.
    show : bool
        Whether to display the plot interactively.
    """
    theta_range = np.linspace(0.0, 0.6, 50)
    phi_range = np.linspace(0.0, 2.0 * np.pi, 40)

    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
    U_array = np.vectorize(lambda th, ph: func([th, ph]))(theta_grid, phi_grid)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = np.column_stack(
        (
            phi_grid.ravel(),
            theta_grid.ravel(),
            U_array.ravel(),
        )
    )
    header = f"apex_pos = {apex_pos}, phi = {phi0}, theta = {theta0}"
    np.savetxt(
        save_dir / f"AFM_PES_2D_{timestamp}.csv",
        data,
        delimiter=",",
        header=header,
    )

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.pcolormesh(
        phi_range, theta_range, U_array.T, shading="auto", cmap="viridis"
    )
    ax.scatter(phi0, abs(theta0), c="r", s=30, zorder=10)
    ax.set_title(f"PES | apex: {apex_pos}")
    fig.tight_layout()
    fig.savefig(save_dir / f"AFM_PES_2D_{timestamp}.png", dpi=300)

    if show:
        plt.show()
    plt.close(fig)


def plot_geometry(
    geom, osc_center, osc_dir, save_path="Geometry.png", show=False
):
    """
    Visualize the geometry and oscillation direction.

    Parameters
    ----------
    geom : object
        Must have a `visualize()` method (e.g. ASE Atoms or custom class).
    osc_center : array-like
        Center position of the oscillator (2D).
    osc_dir : array-like
        Direction vector of oscillator motion (2D).
    save_path : str
        Path to save the output image.
    show : bool
        Whether to display the plot interactively.
    """
    fig = plt.figure()
    geom.visualize()

    x0, y0 = osc_center
    dx, dy = osc_dir
    plt.plot(
        [x0 - 0.5 * dx, x0 + 0.5 * dx],
        [y0 - 0.5 * dy, y0 + 0.5 * dy],
        "-",
        zorder=10,
    )
    plt.plot(x0, y0, ".", zorder=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=800)

    if show:
        plt.show()
    plt.close(fig)
