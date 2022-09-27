#!/usr/bin/env python3
import os
import sys
import argparse
import h5py
import shutil
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from subprocess import run
from multiprocessing import Pool

#####
#
# Steps:
# Identify an hdf5 file to read from OR an odb and an inp file to turn into an hdf5
# If the inp and odb were given process them, throw out the tmp_npz file, as we only need the hdf5, and select a name for this
# Get the desired 3-D coordinates and time step to extract from the hdf5
# Plot the desired section and either show or save the image(s)
#
#####

class Settings:
    pass

VIEWS = {
        # "View: face on top"  : (elev, azim, roll)
        "Top Face: back"                    : (90, 0, 0),
        "Top Face: right"                   : (90, 0, -90),
        "Top Face: left"                    : (90, 0, 90),
        "Top Face: front"                   : (90, 0, 180),
        "Bottom Face: back"                 : (-90, 0, 180),
        "Bottom Face: right"                : (-90, 0, 90),
        "Bottom Face: left"                 : (-90, 0, -90),
        "Bottom Face: front"                : (-90, 0, 0),
        "Left Face: front"                  : (0, 90, -90),
        "Left Face: top"                    : (0, 90, 0),
        "Left Face: back"                   : (0, 90, 90),
        "Left Face: bottom"                 : (0, 90, 180),
        "Right Face: front"                 : (0, -90, 90),
        "Right Face: top"                   : (0, -90, 180),
        "Right Face: back"                  : (0, -90, -90),
        "Right Face: bottom"                : (0, -90, 0),
        "Front Face: right"                 : (0, 0, -90),
        "Front Face: top"                   : (0, 0, 0),
        "Front Face: left"                  : (0, 0, 90),
        "Front Face: bottom"                : (0, 0, 180),
        "Back Face: left"                   : (0, 180, -90),
        "Back Face: top"                    : (0, 180, 0),
        "Back Face: right"                  : (0, 180, 90),
        "Back Face: bottom"                 : (0, 180, 180),
        "Top-Left Edge: top"                : (45, 90, 0),
        "Top-Left Edge: left"               : (45, 90, 180),
        "Top-Right Edge: top"               : (45, -90, 0),
        "Top-Right Edge: right"             : (45, -90, 180),
        "Top-Front Edge: top"               : (45, 0, 0),
        "Top-Front Edge: front"             : (45, 0, 180),
        "Top-Back Edge: top"                : (45, 180, 0),
        "Top-Back Edge: back"               : (45, 180, 180),
        "Bottom-Left Edge: left"            : (-45, 90, 0),
        "Bottom-Left Edge: bottom"          : (-45, 90, 180),
        "Bottom-Right Edge: right"          : (-45, -90, 0),
        "Bottom-Right Edge: bottom"         : (-45, -90, 180),
        "Bottom-Front Edge: front"          : (-45, 0, 0),
        "Bottom-Front Edge: bottom"         : (-45, 0, 180),
        "Bottom-Back Edge: back"            : (-45, 180, 0),
        "Bottom-Back Edge: bottom"          : (-45, 180, 180),
        "Front-Left Edge: left"             : (0, 45, 90),
        "Front-Left Edge: front"            : (0, 45, -90),
        "Front-Right Edge: front"           : (0, -45, 90),
        "Front-Right Edge: right"           : (0, -45, -90),
        "Back-Left Edge: back"              : (0, 135, 90),
        "Back-Left Edge: left"              : (0, 135, -90),
        "Back-Right Edge: right"            : (0, -135, 90),
        "Back-Right Edge: back"             : (0, -135, -90),
        "Top-Right-Front Vertex: right"     : (45, -45, -120),
        "Top-Right-Front Vertex: top"       : (45, -45, 0),
        "Top-Right-Front Vertex: front"     : (45, -45, 120),
        "Top-Left-Front Vertex: front"      : (45, 45, -120),
        "Top-Left-Front Vertex: top"        : (45, 45, 0),
        "Top-Left-Front Vertex: left"       : (45, 45, 120),
        "Top-Right-Back Vertex: right"      : (45, -135, 120),
        "Top-Right-Back Vertex: top"        : (45, -135, 0),
        "Top-Right-Back Vertex: back"       : (45, -135, -120),
        "Top-Left-Back Vertex: back"        : (45, 135, 120),
        "Top-Left-Back Vertex: top"         : (45, 135, 0),
        "Top-Left-Back Vertex: left"        : (45, 135, -120),
        "Bottom-Right-Front Vertex: right"  : (-45, -45, -60),
        "Bottom-Right-Front Vertex: bottom" : (-45, -45, -180),
        "Bottom-Right-Front Vertex: front"  : (-45, -45, 60),
        "Bottom-Left-Front Vertex: bottom"  : (-45, 45, -180),
        "Bottom-Left-Front Vertex: front"   : (-45, 45, -60),
        "Bottom-Left-Front Vertex: left"    : (-45, 45, 60),
        "Bottom-Right-Back Vertex: right"   : (-45, -135, 60),
        "Bottom-Right-Back Vertex: back"    : (-45, -135, -60),
        "Bottom-Right-Back Vertex: bottom"  : (-45, -135, -180),
        "Bottom-Left-Back Vertex: bottom"   : (-45, 135, -180),
        "Bottom-Left-Back Vertex: left"     : (-45, 135, -60),
        "Bottom-Left-Back Vertex: back"     : (-45, 135, 60)
        }    

def main():

    cwd, target_file = process_input()

    x_low, x_high, y_low, y_high, z_low, z_high, time_low, time_high = get_extrema()

    step = get_step()

    # You must always subtract one step size from the max, because of how it is discreteized
    x_high -= step
    y_high -= step
    z_high -= step

    # Adapted from CJ's read_hdf5.py
    coords_df = get_coords(target_file)
    bounded_nodes = list(
            coords_df[
                (coords_df["x"] <= x_high) & (coords_df["x"] >= x_low) &
                (coords_df["y"] <= y_high) & (coords_df["y"] >= y_low) &
                (coords_df["z"] <= z_high) & (coords_df["z"] >= z_low)]["Node Labels"]
            )

    print(f"Extracting Data for {len(bounded_nodes)} nodes")
    with Pool() as pool:
        results = pool.map(read_node_data, zip(bounded_nodes, [target_file for _ in range(len(bounded_nodes))]))

    out_nodes = pd.concat(results)
    print("Filtering by Time")
    out_nodes = out_nodes[(out_nodes["Time"] <= time_high) & (out_nodes["Time"] >= time_low)]

    #TODO Is it worth trying to pre-filter, i.e., keep x and y constant, move z, keep x and z constant, move y, etc.?
    print("Pre-processing data")
    out_nodes_low_time = out_nodes[out_nodes["Time"] == time_low]
    Xs = list()
    Ys = list()
    Zs = list()
    for _, node in out_nodes_low_time.iterrows():
        x = round(node["X"], 5)
        y = round(node["Y"], 5)
        z = round(node["Z"], 5)
        if (x / step == x // step) and (y / step == y // step) and (z / step == z // step):
            Xs.append(x)
            Ys.append(y)
            Zs.append(z)

    Xs = list(dict.fromkeys(Xs))
    Ys = list(dict.fromkeys(Ys))
    Zs = list(dict.fromkeys(Zs))

    Xs.sort()
    Ys.sort()
    Zs.sort()

    x_count = len(Xs)
    y_count = len(Ys)
    z_count = len(Zs)

    Xs.append(Xs[-1] + step)
    Ys.append(Ys[-1] + step)
    Zs.append(Zs[-1] + step)

    x_tick_labels = [format(round(x, 2), ".2f") for x in Xs]
    x_ticks = list(range(len(x_tick_labels)))
    x_offset = int(0 - (Xs[0] / step))

    y_tick_labels = [format(round(y, 2), ".2f") for y in Ys]
    y_ticks = list(range(len(y_tick_labels)))
    y_offset = int(0 - (Ys[0] / step))

    z_tick_labels = [format(round(z, 2), ".2f") for z in Zs]
    z_ticks = list(range(len(z_tick_labels)))
    z_offset = int(0 - (Zs[0] / step))

    print("Results Gathered.")

    results_dir = os.path.join(cwd, "results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Outer while governs user input
    while True:
        elev, azim, roll = get_views()

        user_input = ""
        while user_input.lower() not in ("yes", "y", "no", "n"):
            user_input = input("Would you like to view each timestep as it is plotted? (y/n): ")
            if user_input.lower() in ("yes", "y"):
                show_plots = True
            elif user_input.lower() in ("no", "n"):
                show_plots = False
            else:
                print('Error: Please enter "yes" or "no" or "y" or "n"')

        # out_nodes["Time"] has the time values for each node, we only need one
        # Divide length by len(bounded_nodes), go up to that
        times = out_nodes["Time"]
        final_time_idx = int(len(times) / len(bounded_nodes))
        for current_time in times[:final_time_idx]:
            curr_nodes = out_nodes[times == current_time]
            current_time_name = format(round(current_time, 2), ".2f")
            print(f"Plotting time step {current_time_name}")
            file_name = target_file.split("/")[-1].split(".")[0]
            save_str = f"{results_dir}/{file_name}-{current_time_name}.png"

            x_ind, y_ind, z_ind = np.indices((x_count, y_count, z_count))
            voxels = ((x_ind <= 0 ) | (x_ind >= x_count - 1)) | ((y_ind <= 0) | (y_ind >= y_count - 1)) | ((z_ind <= 0 ) | (z_ind >= z_count - 1))
            colors = np.ndarray(shape=(x_count, y_count, z_count), dtype=object)

            fig = plt.figure(figsize=(19.2, 10.8))
            ax = plt.axes(projection="3d", label=f"{file_name}-{current_time_name}")

            ax.set_box_aspect((x_count, y_count, z_count))
            ax.view_init(elev=elev, azim=azim, roll=roll)

            ax.set_xlabel("x")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)

            ax.set_ylabel("y")
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels)

            ax.set_zlabel("z")
            ax.set_zticks(z_ticks)
            ax.set_zticklabels(z_tick_labels)

            ax.set_title(f"3D Contour, time step: {current_time_name}")

            fig.add_axes(ax, label=f"{file_name}-{current_time_name}")
            for idx, node in curr_nodes.iterrows():
                x = node["X"]
                y = node["Y"]
                z = node["Z"]
                temp = node["Temp"]
                if (x / step == x // step) and (y / step == y // step) and (z / step == z // step):
                    color = "blue"
                    if temp >= 1700:
                        color = "gray"
                    elif temp >= 1500:
                        color = "crimson"
                    elif temp >= 1300:
                        color = "red"
                    elif temp >= 1100:
                        color = "goldenrod"
                    elif temp >= 900:
                        color = "yellow"
                    elif temp >= 700:
                        color = "lime"
                    elif temp >= 500:
                        color = "green"
                    elif temp >= 300:
                        color = "cyan"

                    colors[round(x / step) + x_offset, round(y / step) + y_offset, round(z / step) + z_offset] = mcolors.CSS4_COLORS[color]
                    #colors[round(x / step) + x_offset, round(y / step) + y_offset, round(z / step) + z_offset] = mcolors.CSS4_COLORS["white"]

            ### TESTING
            #  colors[0, 0, 0] = mcolors.CSS4_COLORS["red"]
            #  colors[0, 0, 2] = mcolors.CSS4_COLORS["orange"]
            #  colors[2, 0, 0] = mcolors.CSS4_COLORS["yellow"]
            #  colors[2, 0, 2] = mcolors.CSS4_COLORS["green"]
            #  colors[0, 2, 0] = mcolors.CSS4_COLORS["cyan"]
            #  colors[0, 2, 2] = mcolors.CSS4_COLORS["blue"]
            #  colors[2, 2, 0] = mcolors.CSS4_COLORS["magenta"]
            #  colors[2, 2, 2] = mcolors.CSS4_COLORS["purple"]
            ###

            ax.voxels(voxels, facecolors=colors)

            plt.savefig(save_str)
            if show_plots:
                plt.show()
            plt.close(fig)


def get_extrema():
    while True:
        # Get the desired coordinates and time steps to plot
        extrema = {
                ("lower X", "upper X"): None,
                ("lower Y", "upper Y"): None,
                ("lower Z", "upper Z"): None,
                ("lower Time", "upper Time"): None
                }
        for extremum in extrema:
            extrema[extremum] = process_extrema(extremum)

        x_low, x_high = extrema[("lower X", "upper X")]
        y_low, y_high = extrema[("lower Y", "upper Y")]
        z_low, z_high = extrema[("lower Z", "upper Z")]
        time_low, time_high = extrema[("lower Time", "upper Time")]
        print("SELECTED VALUES:")
        print(f"X    From {x_low} to {x_high}")
        print(f"Y    From {y_low} to {y_high}")
        print(f"Z    From {z_low} to {z_high}")
        print(f"Time From {time_low} to {time_high}")
        print()
        is_correct = input("Are these values correct? (Y/n): ")
        if is_correct.lower() in ("", "y", "yes"):
            return x_low, x_high, y_low, y_high, z_low, z_high, time_low, time_high


def get_step():
    while True:
        try:
            step = float(input("Enter the Default Seed Size of the Mesh: (TODO HOW TO WORD THIS) "))

            print(f"Default Seed Size: {step}")
            is_correct = input("Is this correct? (Y/n): ")
            if is_correct.lower() in ("", "y", "yes"):
                return step
        except ValueError:
            print("Error, Default Seed Size must be a number")


def process_extrema(keys):
    results = [None, None]
    for i, key in enumerate(keys):
        if i % 2 == 1:
            inf_addon = ""
            inf_val = np.inf
        else:
            inf_addon = "negative"
            inf_val = -1 * np.inf
        while True:
            try:
                user_input = input(f"Enter the {key} value you would like to plot (Leave blank for {inf_addon} infinity): ")
                if user_input == "":
                    results[i] = inf_val
                else:
                    results[i] = float(user_input)
                break

            except ValueError:
                print("Error, all selected coordinates and time steps must be numbers")

    return results


def process_input():
    cwd = os.path.dirname(os.path.realpath(__file__))
    input_files = ".hdf5 or (.odb and .inp)"

    parser = argparse.ArgumentParser(description="ODB Extractor and Plotter")
    parser.add_argument(input_files, nargs="*")

    results = vars(parser.parse_args())[input_files]

    if len(results) == 1:
        # Ensure the hdf5 exists
        target_file = ensure_hdf(results[0], cwd)
    elif len(results) == 2:
        # Process the given odb and inp files
        target_file = process_odb(results, cwd)
    else:
        sys.exit("Error: You must supply either a .hdf5 file to read from or a pair of .odb and .inp files to process")

    return cwd, target_file


def get_views():
    global VIEWS
    views_list = list(VIEWS.keys())
    while True:
        print("Please Select a Preset View for your plots: ")
        print('To view all default presets, please enter "list"')
        print ('Or, to specify your own view angle, please enter "custom"')
        print("Important Defaults: Top Face: 4, Right Face: 14, Front Face: 18, Top/Right/Front Isometric: 50")
        user_input = input()
        if user_input.lower() == "list":
            print_views(views_list)
        elif user_input.lower() == "custom":
            return get_custom_views()
        else:
            try:
                user_input = int(user_input)
                if 0 > user_input > (len(views_list) + 1):
                    raise ValueError

                return VIEWS[views_list[user_input - 1]]

            except ValueError:
                print(f'Error: input must be "list," "custom," or an integer between 1 and {len(views_list) + 1}')


def get_custom_views():
    while True:
        while True:
            try:
                user_input = input("Elevation Value to view the plot (Leave Blank for the default): ")
                if user_input == "":
                    elev = "default"
                else:
                    elev = float(user_input)
                break
            except ValueError:
                print("Error, Elevation Value must be a number or left blank")
        while True:
            try:
                user_input = input("Azimuth Value to view the plot (Leave Blank for the default): ")
                if user_input == "":
                    azim = "default"
                else:
                    azim = float(user_input)
                break
            except ValueError:
                print("Error, Azimuth Value must be a number or left blank")
        while True:
            try:
                user_input = input("Roll Value to view the plot (Leave Blank for the default): ")
                if user_input == "":
                    roll = "default"
                else:
                    roll = float(user_input)
                break
            except ValueError:
                print("Error, Roll Value must be a number or left blank")

        print(f"Elevation: {elev}")
        print(f"Azimuth:   {azim}")
        print(f"Roll:      {roll}")
        print()
        is_correct = input("Are these values correct? (Y/n): ")
        if is_correct.lower() in ("", "y", "yes"):
            break

    if elev == "default":
        elev = 30
    if azim == "default":
        azim = -60
    if roll == "default":
        roll = 0
    
    return elev, azim, roll


def print_views(views):
    print("View Index | View Angle: Face on Top")
    for idx, view in enumerate(views):
        print(f"{idx + 1}: {view}")


def ensure_hdf(input_file: str, cwd: str) -> str:
    cwd_file = os.path.join(cwd, input_file)
    hdfs_path = os.path.join(cwd, "hdfs")
    hdfs_path_file = os.path.join(hdfs_path, input_file)
    if not os.path.exists(cwd_file):
        if not os.path.exists(hdfs_path_file):
            sys.exit("Error: .hdf5 file could not be found.")
        return hdfs_path_file
    return cwd_file


def process_odb(input_files: list, cwd: str) -> str:

    output_dir = os.path.join(cwd, "hdfs")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # First, odb_to_npz.py
    odb_file, inp_file = input_files

    filename = odb_file.split(".")[0]
    user_input = input(f"What would you like to name the output file (.hdf5 will be appended automatically). Enter nothing for the default {filename}.hdf5: ")

    if user_input != "":
        filename = user_input

    output_file = f"{filename}.hdf5"

    odb_to_npz_args = ["abq2019", "python", "odb_to_npz.py", odb_file, inp_file]
    run(odb_to_npz_args)
    # By default, the odb_to_npz.py file creates tmp_npz, which we'll use as our goal file
    npz_dir = os.path.join(cwd, "tmp_npz")

    # Adapted from CJ's general purpose npz to hdf code
    filename_list = []
    for root, _, files in os.walk(npz_dir, topdown=True):
        for filename in files:
            filename_list.append(os.path.join(root, filename))

    # Convert to HD5
    os.chdir(output_dir)
    with h5py.File(output_file, "w") as hdf5_file:
        for item in filename_list:
            npz = np.load(item)
            arr = npz[npz.files[0]]
            item_name = os.path.splitext(item)[0].replace(npz_dir, "")
            hdf5_file.create_dataset(item_name, data=arr, compression="gzip")

    os.chdir(cwd)
    if os.path.exists(npz_dir):
        shutil.rmtree(npz_dir)

    return os.path.join(output_dir, output_file)


def get_coords(hdf5_filename: str) -> pd.DataFrame:
    """Gets all coordinates of the HDF5 file related to its nodes."""
    with h5py.File(hdf5_filename, "r") as hdf5_file:
       coords = hdf5_file["node_coords"][:]
       node_labels, x, y, z = np.transpose(coords)
    out_data = pd.DataFrame.from_dict({"Node Labels": node_labels.astype(int), "x": x, "y": y, "z": z})
    return out_data


def read_node_data(data) -> pd.DataFrame:
    """Creates a long format DataFrame with rows being nodes that represent different important information per node."""
    node_label, hdf5_filename = data
    with h5py.File(hdf5_filename, "r") as hdf5_file:
        coords = hdf5_file["node_coords"][:]
        node_coords = coords[np.where(coords[:, 0] == node_label)[0][0]][1:]

        temps = []
        times = []
        temp_steps = hdf5_file["temps"]
        time_steps = hdf5_file["step_frame_times"]
        for i, step in enumerate(temp_steps):
            for frame in temp_steps[step]:
                # Nodes start at 1 not 0
                temps.append(temp_steps[step][frame][node_label - 1])
                times.append(time_steps[step][int(frame.replace("frame_", ""))])
        data_dict = {
                "Node Label": node_label,
                "X": node_coords[0],
                "Y": node_coords[1],
                "Z": node_coords[2],
                "Temp": temps,
                "Time": times
                }

    return pd.DataFrame(data_dict, index=None).sort_values("Time")


if __name__ == "__main__":
    main()
