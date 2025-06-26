import numpy as np
import copy


J_per_eV = 1.6022e-19


def kernelMatrix(x1_vec_0, x2_vec_0, tau):

    x1_vec = np.atleast_3d(x1_vec_0)
    x2_vec = np.atleast_3d(x2_vec_0)

    x1 = np.tile(x1_vec, len(x2_vec))
    x2 = np.tile(x2_vec, len(x1_vec))

    x1 = np.moveaxis(x1, 2, 1)
    x2 = x2.T
    x2 = np.moveaxis(x2, 2, 1)

    x_diff = x1 - x2

    a = np.sum(x_diff**2, axis=2)

    M = np.exp(-0.5 / tau**2 * a)

    return M


class GPR:
    def __init__(self, x, y, tau, sigma):
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
