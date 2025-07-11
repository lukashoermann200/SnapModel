import os
import glob
import re
import numpy as np


def runSnapModel(
    snap_model, StartZ, StopZ, ZStep, osc_dir_0, osc_dir, f0, NSteps
):

    N_ZStep = int(np.round(abs(StartZ - StopZ) / ZStep)) + 1

    apex_height_list = np.linspace(StartZ, StopZ, N_ZStep)
    dissipation = []
    theta_list = []
    phi_list = []
    force = []
    U = []
    U_PES = []

    for apex_height in apex_height_list:
        th, phi = getInitialValuesForPhiAndTheta(osc_dir_0, osc_dir)
        df, Ediss = snap_model.run_oscillation(
            f0, NSteps, apex_height, th, phi
        )

        print(apex_height, df, Ediss, flush=True)

        dissipation.append(Ediss)
        theta_list.append(snap_model.ThVec)
        phi_list.append(snap_model.PhVec)
        force.append(snap_model.FVec)
        U.append(snap_model.UVec)
        U_PES.append(snap_model.UPESVec)

    dissipation = np.array(dissipation)
    theta_list = np.array(theta_list)
    phi_list = np.array(phi_list)
    force = np.array(force)
    U = np.array(U)
    U_PES = np.array(U_PES)

    # write energy dissipation data to file
    output_array = np.vstack(
        (
            apex_height_list,
            theta_list.T,
            phi_list.T,
            force.T,
            U.T,
            U_PES.T,
            dissipation,
        )
    )

    return output_array


def getInitialValuesForPhiAndTheta(osc_dir_0, osc_dir):
    th = 0.0
    phi = np.arccos(osc_dir[0])
    sign = np.sign(np.cross(osc_dir, osc_dir_0)[2])
    phi *= sign

    return th, phi


def saveData(
    tors,
    CO_length,
    horizontal_offset,
    asc_amp,
    angle,
    themal_activation,
    temperature,
    ML,
    NSteps,
    StartZ,
    StopZ,
    ZStep,
    output_array,
    base_dir=None,
):

    N_ZStep = int(np.round(abs(StartZ - StopZ) / ZStep)) + 1

    # write energy dissipation data to file
    if base_dir is None:
        if themal_activation:
            base_dir = "output_TA"
        else:
            base_dir = "output_2D"

    if ML:
        np.save(
            base_dir
            + "/snap_model_ML_TA_tors{:d}_offset{}_amp{}_angle{}_temperature{}".format(
                int(tors * 1e21),
                horizontal_offset,
                asc_amp,
                angle,
                temperature,
            ),
            output_array,
        )
    else:
        np.save(
            base_dir
            + "/snap_model_LJ_TA_tors{:d}_offset{}_amp{}_angle{}_temperature{}".format(
                int(tors * 1e21),
                horizontal_offset,
                asc_amp,
                angle,
                temperature,
            ),
            output_array,
        )


def readAllData(file_name):
    output_array = np.load(file_name)

    r = int((output_array.shape[0] - 2) / 5)

    theta = output_array[1 : r + 1, :]
    phi = output_array[r + 1 : 2 * r + 1, :]
    force = output_array[2 * r + 1 : 3 * r + 1, :]
    U = output_array[3 * r + 1 : 4 * r + 1, :]
    U_PES = output_array[4 * r + 1 : 5 * r + 1, :]

    return theta, phi, force, U, U_PES


def readData(file_name):
    output_array = np.load(file_name)

    z_ind = np.argmax(output_array[-1])

    theta, phi, force, U, U_PES = readAllData(file_name)

    return (
        theta[:, z_ind],
        phi[:, z_ind],
        force[:, z_ind],
        U[:, z_ind],
        U_PES[:, z_ind],
    )


def saveDataAsImage(
    base_dir,
    tors,
    CO_length,
    horizontal_offset,
    asc_amp,
    angle,
    temperature,
    PES_method,
    NSteps,
    StartZ,
    StopZ,
    ZStep,
    output_array,
):

    N_ZStep = int(np.round(abs(StartZ - StopZ) / ZStep)) + 1

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 12))

    bottom_spacing = 0.14
    top_spacing = 0.02
    plot_height = 1 - bottom_spacing - top_spacing
    plot_height_individual = 1

    plot_steps = N_ZStep
    plot_ylabel_setp = int(N_ZStep / 2)

    dissipation = output_array[-1]
    theta_list_index = int((len(output_array) - 2) / 5)
    theta_list = output_array[1:theta_list_index].T

    for i in range(plot_steps):
        y = i * plot_height / plot_steps + bottom_spacing
        dy = plot_height_individual / plot_steps

        ax = fig.add_axes([0.06, y, 0.44, dy])

        apex_pos = asc_amp * np.sin(
            2 * np.pi * np.linspace(0, 1, len(theta_list[i, :]))
        )
        tip_pos = apex_pos + CO_length * np.sin(theta_list[i, :])

        # plt.plot( theta[i]*0, '-k', linewidth=0.5 )
        # plt.plot( theta[i], color='C0' )

        plt.plot(apex_pos, color="k", linewidth=0.5)
        plt.plot(tip_pos, color="C3", linewidth=0.5)

        plt.ylim((-2, 2))

        if i == 0:
            plt.xlabel("oscillation cycle / rad")
            x_pos = np.linspace(0, NSteps - 1, 3)
            x_ticks = ["0", r"$\pi$", r"2$\pi$"]
            plt.xticks(x_pos, x_ticks)
        else:
            plt.xticks([], [])

        if i == plot_ylabel_setp:
            plt.ylabel(r"lateral displacement / a.u.")

        plt.yticks([], [])

    l0 = StopZ - StartZ
    plot_height_corrected = (
        plot_height / plot_steps * (plot_steps - 1)
        + plot_height_individual / plot_steps
    )

    padding = l0 / plot_height_corrected * dy / 2

    y_list = np.linspace(StartZ, StopZ, plot_steps)

    ax0 = fig.add_axes([0.52, bottom_spacing, 0.34, plot_height_corrected])
    ax0.yaxis.tick_right()
    ax0.yaxis.set_label_position("right")

    L = dissipation < 0.0
    dissipation[L] = 0.0

    plt.barh(y_list, dissipation * 1000, align="center", height=0.01)
    plt.xlim((0, 25))
    plt.ylim((StartZ - padding, StopZ + padding))

    plt.xlabel(r"$E_{diss}$ / meV")
    plt.ylabel(r"apex height / $\AA$")

    fig.savefig(
        base_dir
        + "/snap_model_"
        + PES_method
        + "_TA_tors{:d}_offset{}_amp{}_angle{}_temperature{}.png".format(
            int(np.round(tors * 1e21)),
            horizontal_offset,
            asc_amp,
            angle,
            temperature,
        ),
        dpi=600,
    )

    plt.close()


def getEnergyDissipationFromFilesTA(
    output,
    PES_method,
    tors_filter,
    angle_filter,
    amp_filter,
    temperature_filter,
    return_max_energy_dissipation_file=False,
):
    energy_disspation_dict = {}
    max_energy_dissipation_file = None
    max_energy_dissipation = 0

    file_list = glob.glob(output + "/*.npy")
    for file in file_list:
        if PES_method in os.path.basename(file):
            s = "(?<=" + re.escape("amp") + ")"
            amp = re.findall(s + "[0-9\.\-]*", os.path.basename(file))[0]
            amp = np.float64(amp)

            s = "(?<=" + re.escape("tors") + ")"
            tors = re.findall(s + "[0-9\.\-]*", os.path.basename(file))[0]
            tors = np.float64(tors)

            s = "(?<=" + re.escape("offset") + ")"
            offset = re.findall(s + "[0-9\.\-]*", os.path.basename(file))[0]
            offset = np.float64(offset)

            s = "(?<=" + re.escape("angle") + ")"
            angle = re.findall(s + "[0-9\.\-]*", os.path.basename(file))[0]
            angle = np.float64(angle)

            s = "(?<=" + re.escape("temperature") + ")"
            temperature = re.findall(s + "[0-9\.\-]*", os.path.basename(file))[
                0
            ]
            if temperature[-1] == ".":
                temperature = temperature[:-1]
            temperature = np.float64(temperature)

            if (
                not np.float64(tors) == np.float64(tors_filter)
                or not np.float64(angle) == np.float64(angle_filter)
                or not np.float64(amp) == np.float64(amp_filter)
                or not np.float64(temperature)
                == np.float64(temperature_filter)
            ):
                continue

            data = np.load(file)

            for height_ind in range(len(data[-1])):
                params = height_ind

                energy_dissipation = data[-1][height_ind]

                if np.isnan(energy_dissipation):
                    print("energy dissipation nan")
                    continue

                if not params in energy_disspation_dict:
                    energy_disspation_dict[params] = energy_dissipation
                else:
                    energy_disspation_dict[params] = np.max(
                        [energy_disspation_dict[params], energy_dissipation]
                    )

                if energy_dissipation > max_energy_dissipation:
                    max_energy_dissipation_file = file
                    max_energy_dissipation = energy_dissipation

    if return_max_energy_dissipation_file:
        return energy_disspation_dict, max_energy_dissipation_file
    else:
        return energy_disspation_dict


def getEnergyDissipationFromFiles(
    output,
    PES_method,
    max_offset,
    amp_plot,
    min_ind,
    max_ind,
    angle_correction=0.0,
    exclude_string="None",
    params_to_write="tors_angle",
):
    """


    Parameters
    ----------
    output : TYPE
        DESCRIPTION.
    PES_method : TYPE
        DESCRIPTION.
    max_offset : TYPE
        DESCRIPTION.
    amp_plot : TYPE
        DESCRIPTION.
    min_ind : TYPE
        DESCRIPTION.
    max_ind : TYPE
        DESCRIPTION.
    angle_correction : TYPE, optional
        DESCRIPTION. The default is 0.0.
    exclude_string : TYPE, optional
        DESCRIPTION. The default is 'None'.
    params_to_write : str, optional
        Options:
            - tors_angle
            - tors_angle_offset
        The default is 'tors_angle'.

    Returns
    -------
    energy_disspation_dict : TYPE
        DESCRIPTION.

    """
    energy_disspation_dict = {}

    file_list = glob.glob(output + "/*.npy")
    for file in file_list:
        if PES_method in os.path.basename(
            file
        ) and not exclude_string in os.path.basename(file):
            s = "(?<=" + re.escape("amp") + ")"
            amp = re.findall(s + "[0-9\.\-]*", os.path.basename(file))[0]
            amp = np.float64(amp)

            s = "(?<=" + re.escape("tors") + ")"
            tors = re.findall(s + "[0-9\.\-]*", os.path.basename(file))[0]
            tors = np.float64(tors)

            s = "(?<=" + re.escape("offset") + ")"
            offset = re.findall(s + "[0-9\.\-]*", os.path.basename(file))[0]
            offset = np.float64(offset)

            s = "(?<=" + re.escape("angle") + ")"
            angle = re.findall(s + "[0-9\.\-]*", os.path.basename(file))[0]
            if angle[-1] == ".":
                angle = angle[:-1]
            angle = np.float64(angle)

            if np.abs(offset) > max_offset or not amp == amp_plot:
                continue

            if params_to_write == "tors_angle":
                params = (tors, angle - angle_correction)
            elif params_to_write == "tors_angle_offset":
                params = (tors, angle - angle_correction, offset)

            data = np.load(file)
            max_ind_test = np.argmax(data[-1])

            if max_ind_test > max_ind[tors] or max_ind_test < min_ind[tors]:
                max_energy_dissipation = 0.0
            else:
                max_energy_dissipation = np.max(data[-1])

            if not params in energy_disspation_dict:
                energy_disspation_dict[params] = max_energy_dissipation
            else:
                energy_disspation_dict[params] = np.max(
                    [energy_disspation_dict[params], max_energy_dissipation]
                )

    return energy_disspation_dict


def readEnergyDissipationFile(file_name, z_ind="max"):

    PES_method = os.path.basename(file_name)[11:13]

    s = "(?<=" + re.escape("amp") + ")"
    amp = re.findall(s + "[0-9\.\-]*", os.path.basename(file_name))[0]
    amp = np.float64(amp)

    s = "(?<=" + re.escape("tors") + ")"
    tors = re.findall(s + "[0-9\.\-]*", os.path.basename(file_name))[0]
    tors = np.float64(tors)

    s = "(?<=" + re.escape("offset") + ")"
    offset = re.findall(s + "[0-9\.\-]*", os.path.basename(file_name))[0]
    offset = np.float64(offset)

    s = "(?<=" + re.escape("angle") + ")"
    angle = re.findall(s + "[0-9\.\-]*", os.path.basename(file_name))[0]
    if angle[-1] == ".":
        angle = angle[:-1]
    angle = np.float64(angle)

    output_array = np.load(file_name)

    if z_ind == "max":
        z_ind = np.argmax(output_array[-1])
    elif z_ind == "all":
        z_ind = np.array(list(range(len(output_array[-1]))))

    r = int((output_array.shape[0] - 2) / 5)

    theta = output_array[1 : r + 1, z_ind]
    phi = output_array[r + 1 : 2 * r + 1, z_ind]
    force = output_array[2 * r + 1 : 3 * r + 1, z_ind]
    U = output_array[3 * r + 1 : 4 * r + 1, z_ind]
    U_PES = output_array[4 * r + 1 : 5 * r + 1, z_ind]

    return PES_method, amp, tors, offset, angle, theta, phi, force, U, U_PES


def getParametersFromFile(file_name):

    s = "(?<=" + re.escape("amp") + ")"
    amp = re.findall(s + "[0-9\.\-]*", os.path.basename(file_name))[0]
    amp = np.float64(amp)

    s = "(?<=" + re.escape("tors") + ")"
    tors = re.findall(s + "[0-9\.\-]*", os.path.basename(file_name))[0]
    tors = np.float64(tors)

    s = "(?<=" + re.escape("offset") + ")"
    offset = re.findall(s + "[0-9\.\-]*", os.path.basename(file_name))[0]
    offset = np.float64(offset)

    s = "(?<=" + re.escape("angle") + ")"
    angle = re.findall(s + "[0-9\.\-]*", os.path.basename(file_name))[0]
    if angle[-1] == ".":
        angle = angle[:-1]
    angle = np.float64(angle)

    return amp, tors, offset, angle


def read(filename):
    energy_dissipation = []

    with open(filename) as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip().split(" ")

            for l in line:
                if not l == "":
                    energy_dissipation.append(float(l))

    return np.array(energy_dissipation)
