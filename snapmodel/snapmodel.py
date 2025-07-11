import numpy as np
import scipy
import copy
import dfttoolkit.utils.units as units
from snapmodel.utilities import debug_plots
from BADASS.PESUtils import getBarrierHeight


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
        themal_activation=False,
        temperature=5,
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
        self.themal_activation = themal_activation
        self.temperature = temperature
        self.debug_plot_PES = debug_plot_PES

        # plotGeometry(self.energy_model.geom, osc_center, osc_dir)

    def run(self, f0, NSteps_0, apex_height, th, phi):
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
        delta_U_vec = np.zeros([NSteps])

        if self.themal_activation:
            min_apex_phase, max_apex_phase = self.getMaxApexPhase()
            self.min_apex_phase = min_apex_phase
            self.max_apex_phase = max_apex_phase

        # First oscillation for a weird snap
        for i in range(self.init_steps):
            for cnt in range(NSteps):
                t_now = cnt * dt

                apex_pos = self.getApexPos(t_now, T, apex_height)
                U, ThNow, PhiNow = self.AdvTip(apex_pos, ThNow, PhiNow)
                U, ThNow, PhiNow, delta_U = self.forceSnap(
                    t_now, T, apex_height, U, ThNow, PhiNow
                )

        # Then keep oscillating
        for cnt in range(NSteps):
            t_now = cnt * dt

            apex_pos = self.getApexPos(t_now, T, apex_height)
            U, ThNow, PhiNow = self.AdvTip(apex_pos, ThNow, PhiNow)
            U, ThNow, PhiNow, delta_U = self.forceSnap(
                t_now, T, apex_height, U, ThNow, PhiNow
            )

            self.UVec[cnt] = U
            self.ThVec[cnt] = ThNow
            self.PhVec[cnt] = PhiNow
            delta_U_vec[cnt] = delta_U

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

        if self.themal_activation:
            E_diss = self.correctEnergyBySnappingProbability(
                E_diss, delta_U_vec
            )

        # divide by number of cycles to get dissipation per cycle
        df /= self.n_cycles
        E_diss /= self.n_cycles

        return df, E_diss

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

    def forceSnap(self, t_now, T, apex_height, U, theta_0, phi_0):
        """
        Functionality
        -------------
        Checks is a snap has happened and forces the snap if there is a barrier.

        Parameters
        ----------
        t_now : float
            current time in oscillation
        T : float
            total time of osciallation
        apex_height : float
            height of metal apex above bond
        U : float
            energy of CO-tip position before forced snap
        theta_0 : float
            theta of CO-tip position before forced snap
        phi_0 : float
            phi of CO-tip position before forced snap

        Returns
        -------
        U : float
            energy of CO-tip position after forced snap
        theta : float
            theta of CO-tip position after forced snap
        phi : float
            phi of CO-tip position after forced snap
        delta_U : TYPE
            height of barrier that was overcome by forced snapping
        """
        theta = copy.deepcopy(theta_0)
        phi = copy.deepcopy(phi_0)
        delta_U = 0.0

        if self.themal_activation:

            apex_phase = self.getApexPhase(t_now, T)

            apex_pos = self.getApexPos(t_now, T, apex_height)
            CO_pos = self.getCOPos(apex_pos, theta_0, phi_0)
            CO_vec = CO_pos - apex_pos
            CO_side = np.sign(np.dot(CO_vec, self.osc_dir))

            if (
                apex_phase == self.min_apex_phase
                and CO_side > 0
                or apex_phase == self.max_apex_phase
                and CO_side < 0
            ):
                theta = -np.pi * 0.1 * CO_side
                phi = self.Phi_init
                U, theta, phi = self.AdvTip(apex_pos, theta, phi)

                delta_U = self.calculateBarrier(
                    apex_pos, theta_0, phi_0, theta, phi
                )

        return U, theta, phi, delta_U

    def calculateBarrier(self, apex_pos, theta_0, phi_0, theta_1, phi_1):
        """
        Parameters
        ----------
        apex_pos : array
            position of metal apex in Caresian coordinates
        theta_0 : float
            angle theta of first minimum
        phi_0 : float
            angle phi of first minimum
        theta_1 : float
            angle theta of second minimum
        phi_1 : float
            angle phi of second minimum

        Returns
        -------
        delta_U : float
            height of barrier between both minima
        """

        if phi_0 < 0:
            phi_0 += 2.0 * np.pi

        if phi_1 < 0:
            phi_1 += 2.0 * np.pi

        theta_list = np.linspace(-np.pi * 0.25, np.pi * 0.25, 50)
        phi_list = np.linspace(0.0, np.pi * 2.0, 100)

        U_array = np.zeros((50, 100))

        for ind_0, theta in enumerate(theta_list):
            for ind_1, phi in enumerate(phi_list):
                U_array[ind_0, ind_1] = self.getEnergyOfTip(
                    apex_pos, theta, phi
                )

        minimum_0 = (
            np.argmin(abs(theta_list - theta_0)),
            np.argmin(abs(phi_list - phi_0)),
        )
        minimum_1 = (
            np.argmin(abs(theta_list - theta_1)),
            np.argmin(abs(phi_list - phi_1)),
        )

        delta_U = getBarrierHeight(
            U_array, minimum_0, minimum_1, barrier_to_get=0
        )[0]

        print("barrier", U_array[minimum_0], U_array[minimum_1], delta_U)

        return delta_U

    def getCOPos(self, apex_pos, theta, phi):
        # calculate the project of the molecule vector in the xy-plane
        CO_length_xy = self.CO_length * np.sin(theta)

        CO_pos_x = apex_pos[0] + CO_length_xy * np.cos(phi)
        CO_pos_y = apex_pos[1] + CO_length_xy * np.sin(phi)
        CO_pos_z = apex_pos[2] - self.CO_length * np.cos(theta)
        CO_pos = np.array([CO_pos_x, CO_pos_y, CO_pos_z])

        return CO_pos

    def correctEnergyBySnappingProbability(self, E_diss, delta_U_vec):
        p = np.exp(
            -np.max(delta_U_vec)
            / (units.BOLTZMANN_CONSTANT / units.EV_IN_JOULE)
            / self.temperature
        )

        print("snapping probability", p)

        E_diss *= p
        return E_diss

    def AdvTip(self, apex_pos, ThNow, PhiNow):
        if self.dims == 1:
            return self.AdvTip1D(apex_pos, ThNow, PhiNow)
        else:
            return self.AdvTip2D(apex_pos, ThNow, PhiNow)

    def AdvTip1D(self, apex_pos, ThNow, PhiNow):
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
            debug_plots.plot_debug_pes(func, apex_pos, PhiNow)

        # And the energy at this position (why not :-) )
        return func(res.x), res.x[0], PhiNow

    def AdvTip2D(self, apex_pos, ThNow, PhiNow):
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
            debug_plots.plot_debug_pes_2d(func, apex_pos, PhiNow, ThNow)

        return func(res.x), res.x[0], res.x[1]

    def getEnergyOfTip(self, apex_pos, theta, phi):
        CO_pos = self.getCOPos(apex_pos, theta, phi)

        # Energy at a given position is the sum of all atomic contributions
        UH_PES = self.energy_model.getEnergy(CO_pos)

        # And the torsional spring constant
        UH_T = 0.5 * self.tors * theta**2
        UH_T /= J_per_eV  # convert to eV

        UH = UH_PES + UH_T

        return UH
