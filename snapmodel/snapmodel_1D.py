import numpy as np
import scipy
import copy
import aimstools.Units as Units


J_per_eV = 1.6022e-19


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
    ):
        self.run = run
        self.geom = geom
        self.tip = tip

        self.osc_center = osc_center
        self.osc_dir = osc_dir
        self.asc_amp = asc_amp
        self.CO_length = CO_length

        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max
        self.energy_offset = energy_offset

        self.tau = 0.5
        self.sigma = 0.0001
        self.interpolation_grid = 40
        self.getInterpolatedEnergy()

    def getGeometryFromCoords(self, pos):
        curr_tip = copy.deepcopy(self.tip)
        curr_tip.coords += pos
        curr_tip.coords[:, 2] += pos[2] - np.min(curr_tip.coords[:, 2])

        curr_geom = copy.deepcopy(self.geom)
        curr_geom += curr_tip

        return curr_geom

    def getForceOnTip(self, pos):
        dx = self.osc_dir * 0.01
        dz = np.array([0.0, 0.0, 0.01])

        # get forces in horizontal direction
        energy_x0 = self.getEnergy(pos + dx)
        energy_x1 = self.getEnergy(pos - dx)

        force_h = (energy_x0 - energy_x1) / (2 * np.linalg.norm(dx))

        phi = np.arccos(self.osc_dir[0])
        force_x = force_h * np.cos(phi)
        force_y = force_h * np.sin(phi)

        # get forces in vertical rirection
        energy_z0 = self.getEnergy(pos + dz)
        energy_z1 = self.getEnergy(pos - dz)

        force_z = (energy_z0 - energy_z1) / (2 * np.linalg.norm(dz))

        # not sure about sign
        return -force_x, -force_y, -force_z

    def GPR(self, x, y):
        from BADASS.PESUtils import kernelMatrix

        K1 = kernelMatrix(x, x, self.tau)

        self.K1_inv = np.linalg.inv(K1 + np.eye(len(x)) * self.sigma)
        self.x = x
        self.y = y

        ytest = []
        for xtest in x:
            ytest.append(self.getGPRPrediction(np.atleast_2d(xtest)))
        ytest = np.array(ytest)

        y_diff = ytest - y

        print(np.sqrt(y_diff.dot(y_diff)) / len(y), np.std(y))

    def getGPRPrediction(self, x):
        from BADASS.PESUtils import kernelMatrix

        if x[0, 0] > self.x_max:
            x[0, 0] = self.x_max
        if x[0, 0] < self.x_min:
            x[0, 0] = self.x_min
        if x[0, 1] > self.z_max:
            x[0, 1] = self.z_max
        if x[0, 1] < self.z_min:
            x[0, 1] = self.z_min

        K2 = kernelMatrix(x, self.x, self.tau)
        y_test0 = K2.dot(self.K1_inv)

        # energy_penalty = 0
        # if abs(x[0]) > x_min_max:
        #    energy_penalty = np.exp( (x[0]/x_min_max)**2 )

        return float(y_test0.dot(self.y))

    def getInterpolatedEnergy(self):
        x_array_0 = np.linspace(
            self.x_min, self.x_max, self.interpolation_grid
        )
        z_array_0 = np.linspace(
            self.z_min, self.z_max, self.interpolation_grid
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

        self.gpr = self.GPR(x, V_array)

    def getEnergyML(self, pos):

        geom = self.getGeometryFromCoords(pos)

        # energy_offset = -0.113862874736265E+08 -0.308449496249863E+04
        UH = self.run.predict([geom])[0] - self.energy_offset

        return UH

    def getEnergy(self, pos):

        # X = (pos[0] - self.osc_center[0]) / self.osc_dir[0]
        vec = pos - self.osc_center
        X = vec.dot(self.osc_dir)
        Z = vec[2]

        UH = self.getGPRPrediction(np.atleast_2d([X, Z]))

        return UH

    def getForce(self, XPos, YPos, ZPos):
        pos = np.array([XPos, YPos, ZPos])
        return self.getForceOnTip(pos)


class EnergyModelLJ:
    def __init__(self, geom):
        # self.geom = geom
        self.geom = geom.getPeriodicReplica(
            (1, 1, 1), explicit_replications=([-1, 0, 1], [-1, 0, 1], [0])
        )
        self.geom.removeMetalSubstrate()

        print(len(geom))

    def getLJParams(self, species):

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

    def ULJ(self, XPos, YPos, ZPos, LJ_eps, LJ_r):
        # Return the energy at a given location for interaction between two atoms
        rsq = XPos * XPos + YPos * YPos + ZPos * ZPos
        U = LJ_eps * (LJ_r**12 / rsq**6 - 2 * LJ_r**6 / rsq**3)

        return U

    def getEnergy(self, pos):
        # Returns the potential energy via L-J interactions at a given position of the probe particle
        UH = 0

        coords = self.geom.coords
        species = self.geom.species

        for cnt in range(len(coords)):
            Rel = pos - coords[cnt]

            XRel = Rel[0]
            YRel = Rel[1]
            ZRel = Rel[2]

            LJ_eps, LJ_r = self.getLJParams(species[cnt])

            U = self.ULJ(XRel, YRel, ZRel, LJ_eps, LJ_r)

            UH = UH + U

        return UH

    def getForce(self, XPos, YPos, ZPos):
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

            LJ_eps, LJ_r = self.getLJParams(species[cnt])

            rsq = XRel * XRel + YRel * YRel + ZRel * ZRel
            r = np.sqrt(rsq)

            Fmag = (
                12 * LJ_eps * ((LJ_r / r) ** 12) / r
                - 12 * LJ_eps * ((LJ_r / r) ** 6) / r
            )

            FxH = FxH + Fmag * (XRel / r)
            FyH = FyH + Fmag * (YRel / r)
            FzH = FzH + Fmag * (ZRel / r)

        return FxH, FyH, FzH


class DipolModel:
    def __init__(self, cube_file, dipole):
        """
        Parameters
        ----------
        cube_file : TYPE
            aimstool.CubeFile object
        dipole : TYPE
            float in Debye
        """
        self.cube_file = cube_file
        self.dipole = dipole

    def getEnergy(self, pos):
        dz = np.array([0, 0, self.cube_file.dv3])

        potential_0 = self.cube_file.getValueAtPositions(pos)
        potential_1 = self.cube_file.getValueAtPositions(pos + dz)

        electric_field = (potential_0 - potential_1) / self.cube_file.dv3

        "H/A * e*A / e"
        electric_field = electric_field * Units.HARTREE_IN_EV
        # print(dz, electric_field, Units.HARTREE_IN_EV)

        dipole = self.dipole * Units.DEBYE_IN_EANGSTROM
        energy = electric_field * dipole  # / Units.ELEMENTARY_CHARGE
        return energy

    def getForce(self, pos):
        h = 0.0025
        dx = np.array([1.0, 0.0, 0.0]) * h
        dy = np.array([0.0, 1.0, 0.0]) * h
        dz = np.array([0.0, 0.0, 1.0]) * h

        energy_0 = self.getEnergy(pos)
        energy_dx = self.getEnergy(pos + dx)
        energy_dy = self.getEnergy(pos + dy)
        energy_dz = self.getEnergy(pos + dz)

        force_x = (energy_dx - energy_0) / h
        force_y = (energy_dy - energy_0) / h
        force_z = (energy_dz - energy_0) / h

        return force_x, force_y, force_z


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
        dipol_model=None,
        dims=1,
        init_steps=1,
        optimizer="BFGS",
    ):
        self.geom = geom
        self.tip = tip
        self.osc_center = osc_center
        self.osc_dir = osc_dir
        self.asc_amp = asc_amp
        self.CO_length = CO_length
        self.energy_model = energy_model
        self.dipol_model = dipol_model

        # Parameters of the CO tip
        self.tors = tors

        # Sensor parameters
        self.k = k

        self.dims = dims
        self.init_steps = init_steps
        self.optimizer = optimizer

    def runOscillation(self, f0, NSteps, apex_height, th, phi):
        T = 1 / f0
        t0 = 0
        tnow = 0
        dt = T / NSteps
        ThNow = th
        PhiNow = phi

        self.UVec = np.zeros([NSteps])
        self.UPESVec = np.zeros([NSteps])
        self.FVec = np.zeros([NSteps])
        self.ThVec = np.zeros([NSteps])
        self.PhVec = np.zeros([NSteps])
        XVec = np.zeros([NSteps])

        # First oscillation for a weird snap
        for i in range(self.init_steps):
            for cnt in range(NSteps):
                tnow = t0 + cnt * dt

                # Oscillations happen in the x-direction, so calculate the new position
                # If you wanted to integrate tip tilt, here would be the place.
                x = np.sin(2 * np.pi * tnow / T)
                vec_x = self.osc_dir * self.asc_amp * x
                apex_pos = (
                    self.osc_center + np.array([0.0, 0.0, apex_height]) + vec_x
                )

                U, ThNow, PhiNow = self.AdvTip(
                    apex_pos[0], apex_pos[1], apex_pos[2], ThNow, PhiNow
                )

        # Then keep oscillating
        for cnt in range(NSteps):
            tnow = t0 + cnt * dt

            # Oscillations happen in the x-direction, so calculate the new position
            # If you wanted to integrate tip tilt, here would be the place.
            x = np.sin(2 * np.pi * tnow / T)
            vec_x = self.osc_dir * self.asc_amp * x
            apex_pos = (
                self.osc_center + np.array([0.0, 0.0, apex_height]) + vec_x
            )

            XVec[cnt] = apex_pos[0]
            Ynow = apex_pos[1]
            Znow = apex_pos[2]

            U, ThNow, PhiNow = self.AdvTip(
                XVec[cnt], Ynow, Znow, ThNow, PhiNow
            )
            self.UVec[cnt] = U
            self.ThVec[cnt] = ThNow
            self.PhVec[cnt] = PhiNow

            # print(ThNow, PhiNow)

            # Where am I (to get force)
            LatNow = self.CO_length * np.sin(ThNow)
            XNowCO = XVec[cnt] + LatNow * np.cos(PhiNow)
            YNowCO = Ynow + LatNow * np.sin(PhiNow)
            ZNowCO = Znow - self.CO_length * np.cos(ThNow)

            # Calculate the total force on the tip at this position
            Fx, Fy, Fz = self.energy_model.getForce(XNowCO, YNowCO, ZNowCO)

            pos = np.array([XNowCO, YNowCO, ZNowCO])
            UH_PES = self.energy_model.getEnergy(pos)

            if self.dipol_model is not None:
                Fx_dipol, Fy_dipol, Fz_dipol = self.dipol_model.getForce(pos)
                Fx += Fx_dipol
                Fy += Fy_dipol
                Fz += Fz_dipol

                UH_PES_dipol = self.dipol_model.getEnergy(pos)
                UH_PES += UH_PES_dipol

            self.UPESVec[cnt] = UH_PES
            self.FVec[cnt] = self.osc_dir.dot(np.array([Fx, Fy, Fz]))

        dfint = 0
        Edissint = 0

        # That's that! With the forces at each position of the oscillation, we can output the Df and Ediss signals.
        # The calculation for Df was described by Giessibl Appl Phys Lett 78, 123 (2001) - See Eq. 2
        # One description for Ediss can be found in Ondracek et al. Nanotechnology 27, 274005 (2016) - Appendix B
        for cnt in range(NSteps):
            dfint = dfint + self.FVec[cnt] * np.cos(2 * np.pi * f0 * cnt * dt)
            Edissint = Edissint - self.FVec[cnt] * np.cos(
                2 * np.pi * f0 * cnt * dt
            )

        df = -(f0**2 / self.k / self.asc_amp) * dfint * dt
        Ediss = (2 * np.pi * self.asc_amp * f0) * Edissint * dt

        return df, Ediss

    def AdvTip(self, Xnow, Ynow, Znow, ThNow, PhiNow):
        if self.dims == 1:
            return self.AdvTip1D(Xnow, Ynow, Znow, ThNow, PhiNow)
        else:
            return self.AdvTip2D(Xnow, Ynow, Znow, ThNow, PhiNow)

    def AdvTip1D(self, Xnow, Ynow, Znow, ThNow, PhiNow):

        # print('#########################################################\n')

        def func(x):
            LatNow = self.CO_length * np.sin(x[0])
            XNowCO = Xnow + LatNow * np.cos(PhiNow)
            YNowCO = Ynow + LatNow * np.sin(PhiNow)
            ZNowCO = Znow - self.CO_length * np.cos(x[0])

            # Energy at a given position is the sum of all atomic contributions
            pos = np.array([XNowCO, YNowCO, ZNowCO])
            UH_PES = self.energy_model.getEnergy(pos)

            if self.dipol_model is not None:
                UH_PES_dipol = self.dipol_model.getEnergy(pos)
                UH_PES += UH_PES_dipol

            # And the torsional spring constant
            UH_T = 0.5 * self.tors * x[0] ** 2
            UH_T /= J_per_eV

            UH = UH_PES + UH_T
            # print(x[0], x[1], UH_LJ, UH_T)

            return UH

        x0 = np.array([ThNow])
        limits = [(ThNow - 0.5, ThNow + 0.5)]

        res = scipy.optimize.minimize(
            func, x0=x0, bounds=limits, tol=1e-7, method=self.optimizer
        )

        # And the energy at this position (why not :-) )
        return func(res.x), res.x[0], PhiNow

    def AdvTip2D(self, Xnow, Ynow, Znow, ThNow, PhiNow):

        # print('#########################################################\n')

        def func(x):
            LatNow = self.CO_length * np.sin(x[0])
            XNowCO = Xnow + LatNow * np.cos(x[1])
            YNowCO = Ynow + LatNow * np.sin(x[1])
            ZNowCO = Znow - self.CO_length * np.cos(x[0])

            # Energy at a given position is the sum of all atomic contributions
            pos = np.array([XNowCO, YNowCO, ZNowCO])
            UH_PES = self.energy_model.getEnergy(pos)

            if self.dipol_model is not None:
                UH_PES_dipol = self.dipol_model.getEnergy(pos)
                UH_PES += UH_PES_dipol

            # And the torsional spring constant
            UH_T = 0.5 * self.tors * x[0] ** 2
            UH_T /= J_per_eV

            UH = UH_PES + UH_T
            # print(x[0], x[1], UH_LJ, UH_T)

            return UH

        x0 = np.array([ThNow, PhiNow])
        if self.dims == 1:
            limits = [(ThNow - 0.5, ThNow + 0.5), (PhiNow, PhiNow)]
        else:
            limits = [(ThNow - 0.5, ThNow + 0.5), (PhiNow - 0.5, PhiNow + 0.5)]

        res = scipy.optimize.minimize(
            func, x0=x0, bounds=limits, tol=1e-7, method=self.optimizer
        )

        # And the energy at this position (why not :-) )
        return func(res.x), res.x[0], PhiNow

    def getEnergyOfTip(self, Xnow, Ynow, Znow, ThNow, PhiNow):

        x_array = np.linspace(ThNow - 1, ThNow + 1, 50)
        E_array = np.zeros(len(x_array))
        x_tip = np.zeros(len(x_array))
        z_tip = np.zeros(len(x_array))

        for ind, x in enumerate(x_array):
            LatNow = self.CO_length * np.sin(x)
            XNowCO = Xnow + LatNow * np.cos(PhiNow)
            YNowCO = Ynow + LatNow * np.sin(PhiNow)
            ZNowCO = Znow - self.CO_length * np.cos(x)

            # Energy at a given position is the sum of all atomic contributions
            pos = np.array([XNowCO, YNowCO, ZNowCO])
            UH_PES = self.energy_model.getEnergy(pos)

            if self.dipol_model is not None:
                UH_PES_dipol = self.dipol_model.getEnergy(pos)
                UH_PES += UH_PES_dipol

            # And the torsional spring constant
            UH_T = 0.5 * self.tors * x**2
            UH_T /= J_per_eV

            E_array[ind] = UH_PES

            vec = pos - self.osc_center
            x_tip[ind] = vec.dot(self.osc_dir)
            z_tip[ind] = vec[2]

        return x_array, E_array, x_tip, z_tip


if __name__ == "__main__":  # main( int argc, char *argv[] )
    pass
