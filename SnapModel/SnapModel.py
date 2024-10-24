import numpy as np
import scipy
import copy
import aimstools.Units as Units


J_per_eV = 1.6022e-19


class GPR:
    def __init__(self, x, y, tau, sigma):
        from BADASS.PESUtils import kernelMatrix

        K1 = kernelMatrix(x, x, tau)

        self.K1_inv = np.linalg.inv(K1 + np.eye(len(x)) * sigma)
        self.x = x
        self.y = y
        self.tau = tau
        self.sigma = sigma

        ytest = []
        for xtest in x:
            ytest.append(self.getEnergy(np.atleast_2d(xtest)))
        ytest = np.array(ytest)

        y_diff = ytest - y

        print(np.sqrt(y_diff.dot(y_diff)) / len(y), np.std(y), flush=True)

    def getEnergy(self, x):
        from BADASS.PESUtils import kernelMatrix

        K2 = kernelMatrix(x, self.x, self.tau)
        y_test0 = K2.dot(self.K1_inv)

        return float(y_test0.dot(self.y))


class EnergyModelML:
    def __init__(
        self,
        run,
        geom,
        tip,
        osc_center,
        osc_dir,
        asc_amp,
        CO_length,
        x_min,
        x_max,
        z_min,
        z_max,
        energy_offset,
        dims=1,
        interpolation_grid=(40, 10, 40),
    ):
        self.run = run
        self.geom = geom
        self.tip = tip

        self.osc_center = osc_center
        self.osc_dir = osc_dir
        self.osc_dir_normal = np.array([-osc_dir[1], osc_dir[0], 0.0])
        self.osc_dir_normal /= np.linalg.norm(self.osc_dir_normal)
        self.asc_amp = asc_amp
        self.CO_length = CO_length

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = -0.4
        self.y_max = 0.4
        self.z_min = z_min
        self.z_max = z_max
        self.energy_offset = energy_offset
        self.dims = dims

        self.tau = 0.5
        self.sigma = 0.0001
        self.interpolation_grid = interpolation_grid

        if self.dims == 1:
            self.getInterpolatedEnergy()
        else:
            self.getInterpolatedEnergy2D()

    def getGeometryFromCoords(self, pos):
        curr_tip = copy.deepcopy(self.tip)
        curr_tip.coords += pos
        curr_tip.coords[:, 2] += pos[2] - np.min(curr_tip.coords[:, 2])

        curr_geom = copy.deepcopy(self.geom)
        curr_geom += curr_tip

        return curr_geom

    def getGPRPrediction(self, x):
        if x.shape[1] == 2:
            if x[0, 0] > self.x_max:
                x[0, 0] = self.x_max
            if x[0, 0] < self.x_min:
                x[0, 0] = self.x_min
            if x[0, 1] > self.z_max:
                x[0, 1] = self.z_max
            if x[0, 1] < self.z_min:
                x[0, 1] = self.z_min
        else:
            if x[0, 0] > self.x_max:
                x[0, 0] = self.x_max
            if x[0, 0] < self.x_min:
                x[0, 0] = self.x_min
            if x[0, 1] > self.y_max:
                x[0, 1] = self.y_max
            if x[0, 1] < self.y_min:
                x[0, 1] = self.y_min
            if x[0, 2] > self.z_max:
                x[0, 2] = self.z_max
            if x[0, 2] < self.z_min:
                x[0, 2] = self.z_min

        return self.energy_model.getEnergy(x)

    def getInterpolatedEnergy(self):
        x_array_0 = np.linspace(
            self.x_min, self.x_max, self.interpolation_grid[0]
        )
        z_array_0 = np.linspace(
            self.z_min, self.z_max, self.interpolation_grid[2]
        )

        x_array = []
        z_array = []
        V_array = []

        for x in x_array_0:
            for z in z_array_0:
                vec_x = self.osc_dir * x
                tip_pos = self.osc_center + np.array([0.0, 0.0, z]) + vec_x

                V = self.getEnergyML(tip_pos)

                x_array.append(x)
                z_array.append(z)
                V_array.append(V)

        x = np.vstack((x_array, z_array)).T

        self.energy_model = GPR(x, V_array, self.tau, self.sigma)

    def getInterpolatedEnergy2D(self):
        x_array_0 = np.linspace(
            self.x_min, self.x_max, self.interpolation_grid[0]
        )
        y_array_0 = np.linspace(
            self.y_min, self.y_max, self.interpolation_grid[1]
        )
        z_array_0 = np.linspace(
            self.z_min, self.z_max, self.interpolation_grid[2]
        )

        x_array = []
        y_array = []
        z_array = []
        V_array = []

        for x in x_array_0:
            for y in y_array_0:
                for z in z_array_0:
                    vec_x = self.osc_dir * x
                    vec_y = self.osc_dir_normal * y
                    tip_pos = (
                        self.osc_center
                        + vec_x
                        + vec_y
                        + np.array([0.0, 0.0, z])
                    )

                    V = self.getEnergyML(tip_pos)

                    x_array.append(x)
                    y_array.append(y)
                    z_array.append(z)
                    V_array.append(V)

        x = np.vstack((x_array, y_array, z_array)).T

        self.energy_model = GPR(x, V_array, self.tau, self.sigma)

    def getEnergyML(self, pos):

        geom = self.getGeometryFromCoords(pos)

        # energy_offset = -0.113862874736265E+08 -0.308449496249863E+04
        E = self.run.predict([geom])[0]

        # print('getEnergyML', pos, E, flush=True)
        if np.isnan(E):
            raise
        UH = E - self.energy_offset

        return UH

    def getEnergy(self, pos):
        """
        Parameters
        ----------
        pos : np.array
            position in Cartesian coordinates

        pos_gpr : np.array
            position in GPR coordinates

        Returns
        -------
        UH : TYPE
            DESCRIPTION.

        """

        vec = pos - self.osc_center

        if self.dims == 1:
            x_gpr = vec.dot(self.osc_dir)
            z_gpr = vec[2]
            pos_gpr = np.atleast_2d([x_gpr, z_gpr])
        else:
            x_gpr = vec.dot(self.osc_dir)
            y_gpr = vec.dot(self.osc_dir_normal)
            z_gpr = vec[2]
            pos_gpr = np.atleast_2d([x_gpr, y_gpr, z_gpr])

        UH = self.getGPRPrediction(pos_gpr)

        return UH

    def getForce(self, pos):
        if self.dims == 1:
            return self.getForceOnTip(pos)
        else:
            return self.getForceOnTip2D(pos)

    def getForceOnTip(self, pos):
        d_osc_dir = self.osc_dir * 0.01
        dz = np.array([0.0, 0.0, 0.01])

        # get forces in horizontal direction
        energy_x0 = self.getEnergy(pos + d_osc_dir)
        energy_x1 = self.getEnergy(pos - d_osc_dir)
        force_osc_dir = (energy_x0 - energy_x1) / (
            2 * np.linalg.norm(d_osc_dir)
        )

        # get forces in vertical rirection
        energy_z0 = self.getEnergy(pos + dz)
        energy_z1 = self.getEnergy(pos - dz)
        force_z = (energy_z0 - energy_z1) / (2 * np.linalg.norm(dz))

        force = (
            self.osc_dir * force_osc_dir + np.array([0.0, 0.0, 0.01]) * force_z
        )

        # not sure about sign
        return -force

    def getForceOnTip2D(self, pos):
        d_osc_dir = self.osc_dir * 0.01
        d_osc_dir_normal = self.osc_dir_normal * 0.01
        dz = np.array([0.0, 0.0, 0.01])

        # get forces in horizontal direction
        energy_osc_dir_0 = self.getEnergy(pos + d_osc_dir)
        energy_osc_dir_1 = self.getEnergy(pos - d_osc_dir)
        force_osc_dir = (energy_osc_dir_0 - energy_osc_dir_1) / (
            2 * np.linalg.norm(d_osc_dir)
        )

        energy_osc_dir_normal_0 = self.getEnergy(pos + d_osc_dir_normal)
        energy_osc_dir_normal_1 = self.getEnergy(pos - d_osc_dir_normal)
        force_osc_dir_normal = (
            energy_osc_dir_normal_0 - energy_osc_dir_normal_1
        ) / (2 * np.linalg.norm(d_osc_dir_normal))

        # get forces in vertical rirection
        energy_z0 = self.getEnergy(pos + dz)
        energy_z1 = self.getEnergy(pos - dz)

        force_z = (energy_z0 - energy_z1) / (2 * np.linalg.norm(dz))

        force = (
            self.osc_dir * force_osc_dir
            + self.osc_dir_normal * force_osc_dir_normal
            + np.array([0.0, 0.0, 0.01]) * force_z
        )

        # not sure about sign
        return -force


def getLJParams(species):

    # Lennard-Jones parameters
    # Radius of the probe tip (O at the apex) [m]
    r_probe = 1.6612

    # Interaction energy of the probe particle [meV]
    e_probe = 9.106314

    if species == "O":
        e_atom = 9.106314
        r_atom = 1.6612

    elif species == "C":
        e_atom = 3.7292524
        r_atom = 1.908

    else:
        e_atom = 0.6808054
        r_atom = 1.487

    LJ_eps = (
        np.sqrt(e_probe * e_atom) / 1000
    )  # Careful! This is a non-SI value and saved in meV.

    # All distances, however, are saved in SI units
    LJ_r = r_probe + r_atom

    return LJ_eps, LJ_r


class EnergyModelLJ:
    def __init__(self, geom):

        if geom.isPeriodic():
            self.geom = geom.getPeriodicReplica(
                (1, 1, 1), explicit_replications=([-1, 0, 1], [-1, 0, 1], [0])
            )
        else:
            self.geom = geom

        self.geom.removeMetalSubstrate()

        print(len(self.geom), flush=True)

        species = self.geom.species

        self.LJ_eps_vec = np.zeros(len(species))
        self.LJ_r_vec = np.zeros(len(species))

        for ind in range(len(species)):
            LJ_eps, LJ_r = getLJParams(species[ind])
            self.LJ_eps_vec[ind] = LJ_eps
            self.LJ_r_vec[ind] = LJ_r

    def getEnergy(self, pos):
        coords = self.geom.coords

        vec = np.tile(pos, (coords.shape[0], 1)) - coords
        rsq = np.sum(vec**2, axis=1)

        U = self.LJ_eps_vec * (
            self.LJ_r_vec**12 / rsq**6 - 2 * self.LJ_r_vec**6 / rsq**3
        )

        return np.sum(U)

    def getForce(self, pos):
        XPos = pos[0]
        YPos = pos[1]
        ZPos = pos[2]

        # Returns the total force on the probe particle at a given position
        FxH = 0
        FyH = 0
        FzH = 0

        coords = self.geom.coords
        species = self.geom.species

        for cnt in range(len(coords)):
            XRel = XPos - coords[cnt, 0]
            YRel = YPos - coords[cnt, 1]
            ZRel = ZPos - coords[cnt, 2]

            LJ_eps, LJ_r = getLJParams(species[cnt])

            rsq = XRel * XRel + YRel * YRel + ZRel * ZRel
            r = np.sqrt(rsq)

            Fmag = (
                12 * LJ_eps * ((LJ_r / r) ** 12) / r
                - 12 * LJ_eps * ((LJ_r / r) ** 6) / r
            )

            FxH = FxH + Fmag * (XRel / r)
            FyH = FyH + Fmag * (YRel / r)
            FzH = FzH + Fmag * (ZRel / r)

        return np.array([FxH, FyH, FzH])


class EnergyModelPLJ(EnergyModelML):
    def __init__(
        self,
        run,
        geom,
        tip,
        osc_center,
        osc_dir,
        asc_amp,
        CO_length,
        x_min,
        x_max,
        z_min,
        z_max,
        energy_offset,
        lam,
        interpolation_grid=(40, 10, 40),
    ):

        self.run = run
        self.geom = geom
        self.tip = tip

        self.osc_center = osc_center
        self.osc_dir = osc_dir
        self.osc_dir_normal = np.array([-osc_dir[1], osc_dir[0], 0.0])
        self.osc_dir_normal /= np.linalg.norm(self.osc_dir_normal)

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = -0.4
        self.y_max = 0.4
        self.z_min = z_min
        self.z_max = z_max
        self.energy_offset = energy_offset
        self.lam = lam
        self.interpolation_grid = interpolation_grid

        self.x_train, self.y_train = self.getInterpolatedEnergy2D()

        if geom.isPeriodic():
            self.geom = geom.getPeriodicReplica(
                (1, 1, 1), explicit_replications=([-1, 0, 1], [-1, 0, 1], [0])
            )
        else:
            self.geom = geom

        self.geom.removeMetalSubstrate()

        print(len(self.geom), flush=True)

        self.fitModel()

    def getInterpolatedEnergy2D(self):
        x_array_0 = np.linspace(
            self.x_min, self.x_max, self.interpolation_grid[0]
        )
        y_array_0 = np.linspace(
            self.y_min, self.y_max, self.interpolation_grid[1]
        )
        z_array_0 = np.linspace(
            self.z_min, self.z_max, self.interpolation_grid[2]
        )

        x_array = []
        V_array = []

        for x in x_array_0:
            for y in y_array_0:
                for z in z_array_0:
                    vec_x = self.osc_dir * x
                    vec_y = self.osc_dir_normal * y
                    tip_pos = (
                        self.osc_center
                        + vec_x
                        + vec_y
                        + np.array([0.0, 0.0, z])
                    )

                    V = self.getEnergyML(tip_pos)

                    x_array.append(tip_pos)
                    V_array.append(V)

        x_array = np.array(x_array)

        return x_array, V_array

    def fitModel(self):

        species = self.geom.species

        LJ_eps_vec = np.zeros(len(species))
        LJ_r_vec = np.zeros(len(species))

        for ind in range(len(species)):
            LJ_eps, LJ_r = getLJParams(species[ind])
            LJ_eps_vec[ind] = LJ_eps
            LJ_r_vec[ind] = LJ_r

        params_0 = np.hstack(
            (
                LJ_eps_vec * LJ_r_vec**12,
                0.0 * LJ_r_vec,
                0.0 * LJ_r_vec,
                -2.0 * LJ_eps_vec * LJ_r_vec**6,
            )
        )

        N = []
        for ind, x in enumerate(self.x_train):
            base = self.getBase(x)
            N.append(base)

        N = np.array(N)

        K = np.dot(N.T, N)
        K_inv = np.linalg.inv(K + self.lam * np.identity(len(K)))

        L = np.dot(K_inv, N.T)

        y_train_0 = np.dot(N, params_0)

        self.params = params_0 + np.dot(L, self.y_train - y_train_0)

    def getBase(self, pos):
        coords = self.geom.coords

        vec = np.tile(pos, (coords.shape[0], 1)) - coords
        rsq = np.sum(vec**2, axis=1)

        return np.hstack(
            (1.0 / rsq**6, 1.0 / rsq**5, 1.0 / rsq**4, 1.0 / rsq**3)
        )

    def getEnergy(self, pos):
        base = self.getBase(pos)

        return np.dot(self.params, base)


class EnergyModelMockup:
    def __init__(self, geom, osc_dir, LJ_r, LJ_eps, height, sigma):
        self.geom = geom

        self.osc_dir = osc_dir
        self.osc_dir_normal = np.array([-osc_dir[1], osc_dir[0], 0.0])
        self.osc_dir_normal /= np.linalg.norm(self.osc_dir_normal)

        self.LJ_r = LJ_r
        self.LJ_eps = LJ_eps
        self.height = height
        self.sigma = sigma

    def getEnergy(self, pos):

        U_z = self.LJ_eps * (
            self.LJ_r**12 / pos[2] ** 12 - 2 * self.LJ_r**6 / pos[2] ** 6
        )
        U_x = self.height * np.exp(-0.5 * pos[0] ** 2 / self.sigma)
        U_x_scale = 1 / pos[2] ** 12

        U = U_z + (U_x - self.height) * U_x_scale

        # pos_xz = np.array(pos)
        # pos_xz[1] = 0.0
        # rsq = pos_xz.dot(pos_xz)

        # U = self.LJ_eps * ( self.LJ_r**12 / rsq**6 - 2 * self.LJ_r**6 / rsq**3 )

        return U

    def getForce(self, pos):
        d_osc_dir = self.osc_dir * 0.01
        d_osc_dir_normal = self.osc_dir_normal * 0.01
        dz = np.array([0.0, 0.0, 0.01])

        # get forces in horizontal direction
        energy_osc_dir_0 = self.getEnergy(pos + d_osc_dir)
        energy_osc_dir_1 = self.getEnergy(pos - d_osc_dir)
        force_osc_dir = (energy_osc_dir_0 - energy_osc_dir_1) / (
            2 * np.linalg.norm(d_osc_dir)
        )

        energy_osc_dir_normal_0 = self.getEnergy(pos + d_osc_dir_normal)
        energy_osc_dir_normal_1 = self.getEnergy(pos - d_osc_dir_normal)
        force_osc_dir_normal = (
            energy_osc_dir_normal_0 - energy_osc_dir_normal_1
        ) / (2 * np.linalg.norm(d_osc_dir_normal))

        # get forces in vertical rirection
        energy_z0 = self.getEnergy(pos + dz)
        energy_z1 = self.getEnergy(pos - dz)

        force_z = (energy_z0 - energy_z1) / (2 * np.linalg.norm(dz))

        force = (
            self.osc_dir * force_osc_dir
            + self.osc_dir_normal * force_osc_dir_normal
            + np.array([0.0, 0.0, 0.01]) * force_z
        )

        # not sure about sign
        return -force


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

    def correctEnergyBySnappingProbability(self, E_diss, delta_U_vec):
        p = np.exp(
            -np.max(delta_U_vec)
            / (Units.BOLTZMANN_CONSTANT / Units.EV_IN_JOULE)
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
            plotDebugPES(func, apex_pos, PhiNow)

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
            plotDebugPES2D(func, apex_pos, PhiNow, ThNow)

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
