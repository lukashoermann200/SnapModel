import numpy as np
import scipy
import copy
import random
import aimstools.Units as Units
from aimstools import Utilities as ut


J_per_eV = 1.6022e-19


class SnapModel:
    def __init__(
        self,
        geom,
        tip,
        osc_center,
        osc_dir,
        asc_amp,
        CO_length,
        tors,
        k,
        energy_model,
        init_steps=1,
        n_cycles=1,
        optimizer="Langevin",
        temperature=0,
        timestep=10e-15,
        n_steps=1000,
        CO_lifetime=200e-15,
        debug_plot_PES=False,
    ):
        """
        temperature : float
            Temperature in Kelvin.
        timestep : float
            Time step for the Langevin optimisation in seconds.
        n_steps : int
            Number of Langevin steps.
        CO_lifetime : float
            Lifetime of the frustrate translation mode of CO on a metal tip in
            seconds.

        """
        self.geom = geom
        self.tip = tip
        self.osc_center = osc_center
        self.osc_dir = osc_dir
        self.asc_amp = asc_amp
        self.CO_length = CO_length
        self.energy_model = energy_model

        # Parameters of the CO tip
        self.tors = tors

        # Sensor parameters
        self.k = k

        self.init_steps = init_steps
        self.n_cycles = n_cycles
        self.optimizer = optimizer
        self.temperature = temperature
        self.timestep = timestep
        self.n_steps = n_steps
        self.CO_lifetime = CO_lifetime
        self.debug_plot_PES = debug_plot_PES

        # plotGeometry(self.energy_model.geom, osc_center, osc_dir)

    def getApexPhase(self, t_now, T):
        apex_phase = np.sin(2 * np.pi * t_now / T)
        return apex_phase

    def getMaxApexPhase(self):

        NSteps = len(self.UVec)

        min_apex_phase = 1
        max_apex_phase = -1

        for cnt in range(NSteps):
            apex_phase = np.sin(2 * np.pi * cnt / NSteps)

            if apex_phase < min_apex_phase:
                min_apex_phase = apex_phase

            if apex_phase > max_apex_phase:
                max_apex_phase = apex_phase

        return min_apex_phase, max_apex_phase

    def getApexPos(self, t_now, T, apex_height):

        apex_phase = self.getApexPhase(t_now, T)
        apex_vec = self.osc_dir * self.asc_amp * apex_phase
        apex_pos = (
            self.osc_center + np.array([0.0, 0.0, apex_height]) + apex_vec
        )

        return apex_pos

    def getCOPos(self, apex_pos, theta, phi):
        # calculate the project of the molecule vector in the xy-plane
        CO_length_xy = self.CO_length * np.sin(theta)

        CO_pos_x = apex_pos[0] + CO_length_xy * np.cos(phi)
        CO_pos_y = apex_pos[1] + CO_length_xy * np.sin(phi)
        CO_pos_z = apex_pos[2] - self.CO_length * np.cos(theta)
        CO_pos = np.array([CO_pos_x, CO_pos_y, CO_pos_z])

        return CO_pos

    def perturbeTip_OLd(self, apex_pos, theta_0, phi_0):
        """
        Adds a random perturbation to the position of the tip
        """
        U_0 = self.getEnergyOfTip(apex_pos, theta_0, phi_0)
        U = U_0
        theta = theta_0
        phi = phi_0

        if self.temperature > 0:
            delta_theta = random.uniform(-np.pi, np.pi) * 0.5
            delta_phi = random.uniform(-np.pi, np.pi)

            p = random.uniform(0.0, 1.0)

            for scale in np.linspace(0.0, 1.0, 100):
                theta_test = theta_0 + delta_theta * scale
                phi_test = phi_0 + delta_phi * scale
                U_test = self.getEnergyOfTip(apex_pos, theta_test, phi_test)

                k_B = Units.BOLTZMANN_CONSTANT / Units.EV_IN_JOULE
                delta_U = U_test - U_0
                p_test = np.exp(-delta_U / (k_B * self.temperature))

                if p_test > p:
                    U = U_test
                    theta = theta_test
                    phi = phi_test
                else:
                    break

        return U, theta, phi

    def estimateLocalCurvature(self, apex_pos, theta_0, phi_0, h=1e-4):
        U_0 = self.getEnergyOfTip(apex_pos, theta_0, phi_0)

        U_m_theta = self.getEnergyOfTip(apex_pos, theta_0 - h, phi_0)
        U_p_theta = self.getEnergyOfTip(apex_pos, theta_0 + h, phi_0)
        k_theta = (U_p_theta - 2 * U_0 + U_m_theta) / h**2

        U_m_phi = self.getEnergyOfTip(apex_pos, theta_0, phi_0 - h)
        U_p_phi = self.getEnergyOfTip(apex_pos, theta_0, phi_0 + h)
        k_phi = (U_p_phi - 2 * U_0 + U_m_phi) / h**2

        return k_theta, k_phi

    def perturbeTip(self, apex_pos, theta_0, phi_0):
        """
        Thermally perturb tip orientation using Gaussian fluctuations
        derived from harmonic approximation around (theta_0, phi_0).
        """
        # Estimate local curvature (can be done analytically or numerically)
        k_theta, k_phi = self.estimateLocalCurvature(apex_pos, theta_0, phi_0)

        k_B = Units.BOLTZMANN_CONSTANT / Units.EV_IN_JOULE
        sigma_theta = np.sqrt(k_B * self.temperature / k_theta)
        sigma_phi = np.sqrt(k_B * self.temperature / k_phi)

        if sigma_theta <= 0:
            sigma_theta = 0
        if sigma_phi <= 0:
            sigma_phi = 0

        delta_theta = np.random.normal(loc=0.0, scale=sigma_theta)
        delta_phi = np.random.normal(loc=0.0, scale=sigma_phi)

        if np.isnan(delta_theta) or np.isinf(delta_theta):
            delta_theta = 0
        if np.isnan(delta_phi) or np.isinf(delta_phi):
            delta_phi = 0

        theta_new = theta_0 + delta_theta
        phi_new = phi_0 + delta_phi
        U_new = self.getEnergyOfTip(apex_pos, theta_new, phi_new)

        return U_new, theta_new, phi_new

    def runOscillation(self, f0, NSteps_0, apex_height, th, phi):
        self.Th_init = copy.deepcopy(th)
        self.Phi_init = copy.deepcopy(phi)
        ThNow = th
        PhiNow = phi
        T = 1 / f0
        dt = T / NSteps_0

        NSteps = self.n_cycles * NSteps_0

        self.UVec = np.zeros([NSteps])
        self.UPESVec = np.zeros([NSteps])
        self.FVec = np.zeros([NSteps])
        self.ThVec = np.zeros([NSteps])
        self.PhVec = np.zeros([NSteps])

        # First oscillation for a weird snap
        for i in range(self.init_steps):
            for cnt in range(NSteps):
                t_now = cnt * dt

                apex_pos = self.getApexPos(t_now, T, apex_height)
                # U, ThNow, PhiNow = self.perturbeTip(apex_pos, ThNow, PhiNow)
                U, ThNow, PhiNow = self.advanceTip(apex_pos, ThNow, PhiNow)

                # self.advanceTip2D(apex_pos, ThNow, PhiNow)

        # Then keep oscillating
        for cnt in range(NSteps):
            t_now = cnt * dt

            apex_pos = self.getApexPos(t_now, T, apex_height)
            # U, ThNow, PhiNow = self.perturbeTip(apex_pos, ThNow, PhiNow)
            U, ThNow, PhiNow = self.advanceTip(apex_pos, ThNow, PhiNow)

            self.UVec[cnt] = U
            self.ThVec[cnt] = ThNow
            self.PhVec[cnt] = PhiNow

            # Where am I (to get force)
            CO_pos = self.getCOPos(apex_pos, ThNow, PhiNow)

            # Calculate the total force and energy on the tip at this position
            force_PES = self.energy_model.getForce(CO_pos)
            UH_PES = self.energy_model.getEnergy(CO_pos)

            self.UPESVec[cnt] = UH_PES
            self.FVec[cnt] = self.osc_dir.dot(force_PES)

        dfint = 0
        Edissint = 0

        # That's that! With the forces at each position of the oscillation, we can output the Df and Ediss signals.
        # The calculation for Df was described by Giessibl Appl Phys Lett 78, 123 (2001) - See Eq. 2
        # One description for Ediss can be found in Ondracek et al. Nanotechnology 27, 274005 (2016) - Appendix B
        for cnt in range(NSteps):
            dfint = dfint - self.FVec[cnt] * np.sin(2 * np.pi * f0 * cnt * dt)
            Edissint = Edissint - self.FVec[cnt] * np.cos(
                2 * np.pi * f0 * cnt * dt
            )

        df = -(f0**2 / self.k / self.asc_amp) * dfint * dt
        E_diss = (2 * np.pi * self.asc_amp * f0) * Edissint * dt

        # divide by number of cycles to get dissipation per cycle
        df /= self.n_cycles
        E_diss /= self.n_cycles

        return df, E_diss

    def correctEnergyBySnappingProbability(self, E_diss, delta_U_vec):
        p = np.exp(
            -np.max(delta_U_vec)
            / (Units.BOLTZMANN_CONSTANT / Units.EV_IN_JOULE)
            / self.temperature
        )

        print("snapping probability", p)

        E_diss *= p
        return E_diss

    def advanceTip(self, apex_pos, ThNow, PhiNow):
        if self.optimizer == "Langevin":
            return self.advanceTip_Langevin(apex_pos, ThNow, PhiNow)
        else:
            return self.advanceTip2D(apex_pos, ThNow, PhiNow)

    def advanceTip2D(self, apex_pos, theta, phi):
        """
        Functionality
        -------------
        Optimises (with respect to the energy) the deflection angles theta and
        phi of the CO-molecule for a given postion of the metal apex
        """

        def func(x):
            """
            Functionality
            -------------
            Gives the energy of the CO-molecule at a given apex postions and
            given deflection angles theta and phi; The energy is the sum of the
            component from the PES and the component from the torsion spring

            x : np.array
                [theta, phi] of the CO-molecule
            """
            theta = x[0]
            phi = x[1]

            return self.getEnergyOfTip(apex_pos, theta, phi)

        x0 = np.array([theta, phi])
        limits = [(theta - 2.0, theta + 2.0), (phi - 2.0, phi + 2.0)]

        res = scipy.optimize.minimize(
            func,
            x0=x0,
            bounds=limits,
            tol=1e-7,
        )

        if self.debug_plot_PES:
            plotDebugPES2D(func, apex_pos, theta, phi)

        print(theta, phi, res.x[0], res.x[1])

        return func(res.x), res.x[0], res.x[1]

    def advanceTip_Langevin(
        self,
        apex_pos,
        theta,
        phi,
    ):
        """
        Functionality
        -------------
        Langevin dynamics for rotational motion (theta, phi) of a CO molecule.

        Parameters
        ----------
        apex_pos : ndarray
            Current position of the metal tip apex.
        theta, phi : float
            Initial angles (theta, phi) of the CO molecule in radians.

        """
        kB_eV = 8.617333262145e-5  # eV/K

        # Initialize angular positions and angular velocities
        x = np.array([theta, phi], dtype=float)
        v = np.zeros(2)

        # Moment of inertia of CO molecule
        I_0 = self.getMomentOfInertia()  # kg·m²
        I = I_0 / J_per_eV  # convert to eV·s²

        gamma = I / self.CO_lifetime

        # Langevin noise term scaling (in eV)
        sqrt_term = np.sqrt(
            2 * gamma * kB_eV * self.temperature / self.timestep
        )

        traj = []

        for step in range(self.n_steps):
            tau = self.getTorqueOnTip(apex_pos, x[0], x[1])  # torque in eV/rad

            # stochastic force in eV/rad
            noise = np.random.normal(0.0, 1.0, size=2) * sqrt_term

            # Langevin equation for angular motion
            a = (tau - gamma * v + noise) / I
            v += a * self.timestep
            x += v * self.timestep
            x[1] %= 2 * np.pi

            traj.append(x.copy())

        x_avg = np.mean(traj[int(self.n_steps / 4 * 3) :], axis=0)

        return (
            self.getEnergyOfTip(apex_pos, x_avg[0], x_avg[1]),
            x_avg[0],
            x_avg[1],
        )

    def getMomentOfInertia(self):
        """
        Functionality
        -------------
        Calculates the moment of inertia in kg/m^2 of the CO tip.

        """
        d_C = (
            self.CO_length
            - np.abs(self.tip.coords[0, 2] - self.tip.coords[1, 2])
        ) * 1e-10
        d_O = self.CO_length * 1e-10

        mass_C = ut.ATOMIC_MASSES[6] * Units.ATOMIC_MASS_IN_KG
        mass_O = ut.ATOMIC_MASSES[8] * Units.ATOMIC_MASS_IN_KG

        return mass_C * d_C**2 + mass_O * d_O**2

    def getEnergyOfTip(self, apex_pos, theta, phi):
        """
        Functionality
        -------------
        Determine energy of tip

        Returns
        -------
        UH : float
            energy of tip in eV
        """
        CO_pos = self.getCOPos(apex_pos, theta, phi)

        # Energy at a given position is the sum of all atomic contributions
        UH_PES = self.energy_model.getEnergy(CO_pos)

        # And the torsional spring constant
        UH_T = 0.5 * self.tors * theta**2
        UH_T /= J_per_eV  # convert to eV

        UH = UH_PES + UH_T

        return UH

    def getTorqueOnTip(self, apex_pos, theta, phi):
        """
        Functionality
        -------------
        Determine the torque on the tip in sperical coordiantes.

        Returns
        -------
        F : np.array
            torque on tip in eV/rad
        """
        CO_pos = self.getCOPos(apex_pos, theta, phi)

        # Energy at a given position is the sum of all atomic contributions
        F_pes = self.energy_model.getForce(CO_pos)

        tau_pes = self.convertForceToTorque(F_pes, theta, phi)

        # And the torsional spring constant
        tau_spring = -self.tors * theta  # J/rad
        tau_spring /= J_per_eV  # convert to eV/rad

        F = tau_pes + np.array([tau_spring, 0.0])

        return F

    def convertForceToTorque(self, F_cart, theta, phi):
        r_vec = self.CO_length * np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                -np.cos(theta),
            ]
        )
        torque_vec = np.cross(r_vec, F_cart)

        # Convert to troques
        r_phi = np.array([np.cos(phi), np.sin(phi), 0.0])
        e_tau_phi = np.array([0.0, 0.0, 1.0])
        e_tau_theta = np.cross(r_phi, e_tau_phi)

        # Project torque onto the angular directions
        tau_theta = np.dot(torque_vec, e_tau_theta)
        tau_phi = np.dot(torque_vec, e_tau_phi)

        return np.array([tau_theta, tau_phi])


def plotDebugPES(func, apex_pos, phi):
    import matplotlib.pyplot as plt
    from datetime import datetime

    x_array = np.linspace(-0.5, 0.5, 50)
    U_array = np.zeros_like(x_array)

    for ind, x in enumerate(x_array):
        U_array[ind] = func(np.array([x]))

    data = np.vstack((x_array, U_array))

    np.savetxt(f"AFM_PES_{str(datetime.now())}.csv", data.T, delimiter=",")

    fig = plt.figure()
    plt.title(str(apex_pos) + " " + str(phi / np.pi * 180))
    plt.plot(x_array, U_array)
    plt.xlabel(r"$\theta$ / rad")
    plt.ylabel("energy / eV")
    fig.savefig("AFM_PES_" + str(datetime.now()) + ".png")
    plt.close(fig)


def plotDebugPES2D(func, apex_pos, theta, phi):
    import matplotlib.pyplot as plt
    from datetime import datetime

    phi_debug = np.linspace(0, 2.0 * np.pi, 40)
    theta_debug = np.linspace(0.0, 0.6, 50)

    x_array = np.zeros((len(phi_debug), len(theta_debug)))
    y_array = np.zeros((len(phi_debug), len(theta_debug)))
    U_array = np.zeros((len(phi_debug), len(theta_debug)))

    for ind_1, x in enumerate(phi_debug):
        for ind_2, y in enumerate(theta_debug):
            x_array[ind_1, ind_2] = x
            y_array[ind_1, ind_2] = y
            U_array[ind_1, ind_2] = func(np.array([y, x]))

    data = np.vstack((x_array.flatten(), y_array.flatten(), U_array.flatten()))

    header = (
        f"apex_pos = {str(apex_pos)}\nphi = {str(phi)}\ntheta = {str(theta)}"
    )

    np.savetxt(
        f"AFM_PES_{str(datetime.now())}.csv",
        data.T,
        delimiter=",",
        header=header,
    )

    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    plt.title(str(apex_pos))

    ax.pcolormesh(phi_debug, theta_debug, U_array.T, edgecolors="face")

    if theta < 0:
        theta = abs(theta)
        phi += np.pi

    ax.scatter(phi, theta, c="r", zorder=10000000)

    fig.savefig("AFM_PES_" + str(datetime.now()) + ".png")
    plt.close(fig)


def plotGeometry(geom, osc_center, osc_dir):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    geom.visualize()
    plt.plot(
        [osc_center[0] - 0.5 * osc_dir[0], osc_center[0] + 0.5 * osc_dir[0]],
        [osc_center[1] - 0.5 * osc_dir[1], osc_center[1] + 0.5 * osc_dir[1]],
        "-",
        zorder=100000000,
    )
    plt.plot(
        [osc_center[0], osc_center[0]],
        [osc_center[1], osc_center[1]],
        ".",
        zorder=100000000,
    )

    fig.savefig("Geometry.png", dpi=800)
    plt.close()


def getInitialValuesForPhiAndTheta(osc_dir_0, osc_dir):
    th = 0.0
    phi = np.arccos(osc_dir[0])
    sign = np.sign(np.cross(osc_dir, osc_dir_0)[2])
    phi *= sign

    return th, phi


if __name__ == "__main__":  # main( int argc, char *argv[] )
    pass
