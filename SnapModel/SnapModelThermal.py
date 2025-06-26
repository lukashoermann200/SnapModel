import numpy as np
import scipy
import copy
import random
import aimstools.Units as Units


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
        dims=1,
        init_steps=1,
        n_cycles=1,
        optimizer="BFGS",
        tol=1e-7,
        temperature=0,
        debug_plot_PES=False,
    ):

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

        self.dims = dims
        self.init_steps = init_steps
        self.n_cycles = n_cycles
        self.optimizer = optimizer
        self.tol = tol
        self.temperature = temperature
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

    def perturbeTip(self, apex_pos, theta_0, phi_0):
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

            # print(k_B, U_test, theta, theta_0, phi, phi_0)

        return U, theta, phi

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
                U, ThNow, PhiNow = self.perturbeTip(apex_pos, ThNow, PhiNow)
                U, ThNow, PhiNow = self.advanceTip(apex_pos, ThNow, PhiNow)

        # Then keep oscillating
        for cnt in range(NSteps):
            t_now = cnt * dt

            apex_pos = self.getApexPos(t_now, T, apex_height)
            U, ThNow, PhiNow = self.perturbeTip(apex_pos, ThNow, PhiNow)
            U, ThNow, PhiNow = self.advanceTip(apex_pos, ThNow, PhiNow)
            # U, ThNow, PhiNow = self.perturbeTip(apex_pos, ThNow, PhiNow)

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
        if self.dims == 1:
            return self.advanceTip1D(apex_pos, ThNow, PhiNow)
        else:
            return self.advanceTip2D(apex_pos, ThNow, PhiNow)

    def advanceTip1D(self, apex_pos, ThNow, PhiNow):
        """
        Functionality
        -------------
        Optimises (with respect to the energy) the deflection angle theta of
        the CO-molecule for a given postion of the metal apex
        """

        def func(x):
            """
            Functionality
            -------------
            Gives the energy of the CO-molecule at a given apex postions and given
            deflection angles theta and phi; The energy is the sum of the component
            from the PES and the component from the torsion spring

            x : np.array
                angle theta of the CO-molecule
            """
            theta = x[0]

            return self.getEnergyOfTip(apex_pos, theta, PhiNow)

        x0 = np.array([ThNow])
        limits = [(ThNow - 2.0, ThNow + 2.0)]

        res = scipy.optimize.minimize(
            func, x0=x0, bounds=limits, tol=self.tol, method=self.optimizer
        )

        if self.debug_plot_PES:
            plotDebugPES(func, apex_pos, PhiNow)

        # And the energy at this position (why not :-) )
        return func(res.x), res.x[0], PhiNow

    def advanceTip2D(self, apex_pos, ThNow, PhiNow):
        """
        Functionality
        -------------
        Optimises (with respect to the energy) the deflection angles theta and phi of
        the CO-molecule for a given postion of the metal apex
        """

        def func(x):
            """
            Functionality
            -------------
            Gives the energy of the CO-molecule at a given apex postions and given
            deflection angles theta and phi; The energy is the sum of the component
            from the PES and the component from the torsion spring

            x : np.array
                [theta, phi] of the CO-molecule
            """
            theta = x[0]
            phi = x[1]

            return self.getEnergyOfTip(apex_pos, theta, phi)

        x0 = np.array([ThNow, PhiNow])
        limits = [(ThNow - 2.0, ThNow + 2.0), (PhiNow - 2.0, PhiNow + 2.0)]

        res = scipy.optimize.minimize(
            func, x0=x0, bounds=limits, tol=self.tol, method=self.optimizer
        )

        if self.debug_plot_PES:
            plotDebugPES2D(func, apex_pos, PhiNow, ThNow)

        return func(res.x), res.x[0], res.x[1]

    def getEnergyOfTip(self, apex_pos, theta, phi):
        """
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


def plotDebugPES(func, apex_pos, PhiNow):
    import matplotlib.pyplot as plt
    from datetime import datetime

    x_array = np.linspace(-0.5, 0.5, 50)
    U_array = np.zeros_like(x_array)

    for ind, x in enumerate(x_array):
        U_array[ind] = func(np.array([x]))

    data = np.vstack((x_array, U_array))

    np.savetxt(f"AFM_PES_{str(datetime.now())}.csv", data.T, delimiter=",")

    fig = plt.figure()
    plt.title(str(apex_pos) + " " + str(PhiNow / np.pi * 180))
    plt.plot(x_array, U_array)
    plt.xlabel(r"$\theta$ / rad")
    plt.ylabel("energy / eV")
    fig.savefig("AFM_PES_" + str(datetime.now()) + ".png")
    plt.close(fig)


def plotDebugPES2D(func, apex_pos, phi, theta):
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
