#!/usr/bin/env python3
import os
import sys
import argparse
import h5py
import shutil
import json
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

class CustomAxis:
    def __init__(self):
        self.name = ""

        self.low = None
        self.high = None
        self.vals = None
        self.size = None


class State:

    def __init__(self):

        self.cwd = os.getcwd()
        self.target_file = None
        self.target_file_name = None

        self.results_dir = self.make_results_dir()

        self.x = CustomAxis()
        self.y = CustomAxis()
        self.z = CustomAxis()
        self.time_low = None
        self.time_high = None
        self.mesh_seed_size = None

        self.main_loop = True
        self.show_plots = True

        self.bounded_nodes = None
        self.bounded_nodes_size = None

        self.out_nodes = None
        self.out_nodes_low_time = None

        self.loaded = False

        self.colormap_str = "turbo"
        self.colormap = None
        self.meltpoint = None

        self.views = self.load_json(os.path.join(os.path.join(self.cwd, "views"), "views.json"))

        self.views_list = list(self.views.keys())

        self.angle = self.views_list[49]
        self.elev = self.views[self.angle][0]
        self.azim = self.views[self.angle][1]
        self.roll = self.views[self.angle][2]


        self. help_menu = """ODBPlotter Help:
help             -- Show this menu
quit, exit, q    -- Exit the ODBPlotter
select           -- Select an hdf5 file (or generate an hdf5 file from a pair of .odb and .inp files)
extrema, range   -- Set the upper and lower x, y, and z bounds for plotting
time             -- Set the upper and lower time bounds
process          -- Actually load the selected data from the file set in select
angle            -- Update the viewing angle
show-all         -- Toggle if each time step will be shown in te matplotlib interactive viewer
plot, show       -- Plot each selected timestep
state, settings  -- Show the current state of the settings of the plotter"""


    def __str__(self):
        return f"""X Range:                 {self.x.low} to {self.x.high - self.mesh_seed_size if self.x.high is not None and self.mesh_seed_size is not None else "not set"}
Y Range:                 {self.y.low} to {self.y.high - self.mesh_seed_size if self.y.high is not None and self.mesh_seed_size is not None else "not set"}
Z Range:                 {self.z.low} to {self.z.high - self.mesh_seed_size if self.z.high is not None and self.mesh_seed_size is not None else "net set"}
Time Range:              {self.time_low} to {self.time_high}
Seed Size of the Mesh:   {self.mesh_seed_size}
View Angle:              {self.angle}
View Elevation:          {self.elev}
View Azimuth:            {self.azim}
View Roll:               {self.roll}

Data loaded into memory: {'Yes' if self.loaded else 'No'}

Is each time-step being shown in the matplotlib interactive viewer? {'Yes' if self.show_plots else 'No'}"""

    def make_results_dir(self):
        results_dir = os.path.join(self.cwd, "results")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        return results_dir
    
    def set_mesh_seed_size(self, seed):
        self.mesh_seed_size = seed

        #  if self.x.high is not None:
        #      self.x.high += self.mesh_seed_size
        #
        #  if self.y.high is not None:
        #      self.y.high += self.mesh_seed_size
        #
        #  if self.z.high is not None:
        #      self.z.high += self.mesh_seed_size

    def set_x_extrema(self, low, high):
        self.x.low = low
        #  if self.mesh_seed_size is not None:
        #      self.x.high = (high + self.mesh_seed_size)
        #  else:
        #      self.x.high = high
        self.x.high = high

    def set_y_extrema(self, low, high):
        self.y.low = low
        #  if self.mesh_seed_size is not None:
        #      self.y.high = (high + self.mesh_seed_size)
        #  else:
        #      self.y.high = high
        self.y.high = high

    def set_z_extrema(self, low, high):
        self.z.low = low
        #  if self.mesh_seed_size is not None:
        #      self.z.high = (high + self.mesh_seed_size)
        #  else:
        #      self.z.high = high
        self.z.high = high

    def pre_process_data(self):
        self.out_nodes_low_time = self.out_nodes[self.out_nodes["Time"] == self.time_low]
        self.x.vals = list()
        self.y.vals = list()
        self.z.vals = list()
        for _, node in self.out_nodes_low_time.iterrows():
            _x = round(node["X"], 5)
            _y = round(node["Y"], 5)
            _z = round(node["Z"], 5)
            if (_x % self.mesh_seed_size == 0) and (_y % self.mesh_seed_size == 0) and (_z % self.mesh_seed_size == 0):
                self.x.vals.append(_x)
                self.y.vals.append(_y)
                self.z.vals.append(_z)

        # Makes these in-order lists of unique values
        self.x.vals = list(dict.fromkeys(self.x.vals))
        self.y.vals = list(dict.fromkeys(self.y.vals))
        self.z.vals = list(dict.fromkeys(self.z.vals))

        self.x.vals.sort()
        self.y.vals.sort()
        self.z.vals.sort()

        self.x.vals = np.asarray(self.x.vals)
        self.y.vals = np.asarray(self.y.vals)
        self.z.vals = np.asarray(self.z.vals)

        self.x.size = len(self.x.vals)
        self.y.size = len(self.y.vals)
        self.z.size = len(self.z.vals)

        self.loaded = True

    def load_json(self, file):
        with open(file, "r") as o_file:
            return json.load(o_file)

    def select_colormap(self):
        norm = mcolors.Normalize(0, self.meltpoint)
        self.colormap = plt.cm.ScalarMappable(norm=norm, cmap=self.colormap_str)
        self.colormap.set_array([])


def main():

    state = State()

    # TODO Process input json file and/or cli switches here
    # process_input(state)

    # Outer while governs user input
    user_input = ""
    print("ODBPlotter v.0.1")
    while state.main_loop:
        try:
            print()
            user_input = input("> ")
            user_input = user_input.strip().lower()

            if user_input in ("exit", "quit", "q"):
                state.main_loop = False

            elif user_input in ("select", ):
                select_files(state)
                print(f"Target .hdf5 file: {state.target_file}")

            elif user_input in ("seed", "mesh", "step"):
                set_seed_size(state)
                print(f"Seed size set to: {state.mesh_seed_size}")

            elif user_input in ("extrema", "range"):
                get_extrema(state)
                print(f"Physical Range values updated")

            elif user_input in ("time", ):
                set_time(state)
                print(f"Time Range values updated")

            elif user_input in ("process", ):
                load_hdf(state)

            elif user_input in ("angle", ):
                get_views(state)
                print(f"Angle Updated")

            elif user_input in ("show-all", ):
                state.show_plots = not state.show_plots
                print(f"Plots will now {'BE' if state.show_plots else 'NOT BE'} shown")

            elif user_input in ("plot", "show"):
                plot_time_range(state)

            elif user_input in ("state", "settings"):
                print(state)

            elif user_input in ("help", ):
                print(state.help_menu)

            else:
                print('Invalid option. Use "help" to see available options')

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt Received, returning to main menu")
            print('(From this menu, use the "exit" command to exit, or Control-D/EOF)')

        except EOFError:
            print("\nExiting")
            state.main_loop = False


def confirm(message, default=None):
    check_str = "Is this correct (y/n)? "
    yes_vals = ("yes", "y")
    no_vals = ("no", "n")
    if isinstance(default, str):
        if default.lower() in yes_vals:
            yes_vals = ("yes", "y", "")
            check_str = "Is this correct (Y/n)? "
        elif default.lower() in no_vals:
            no_vals = ("no", "n", "")
            check_str = "Is this correct (y/N)? "

    while True:
        print(message)
        user_input = input(check_str)
        if user_input.lower() in yes_vals:
            return True
        elif user_input.lower() in no_vals:
            return False
        else:
            print("Error: invalid input")


def select_files(state):
    odb_options = ("odb", ".odb")
    hdf_options = (".hdf", "hdf", ".hdf5", "hdf5", "hdfs", ".hdfs")

    while True:
        user_input = input('Please enter either "hdf" if you plan to open .hdf5 file or "odb" if you plan to open a .odb file: ')
        user_input = user_input.strip().lower()

        if user_input in odb_options or user_input in hdf_options:
            if(confirm(f"You entered {user_input}", "yes")):
                break

        else:
            print("Error: invalid input")

    if user_input in odb_options:
        # process odb
        while True:
            user_input = input("Please enter the path of the odb file: ")
            if(confirm(f"You entered {user_input}", "yes")):
                odb = user_input
                break

        while True:
            user_input = input(f"Please enter the path of the inp file for {odb}: ")
            if(confirm(f"You entered {user_input}", "yes")):
                inp = user_input
                break

        state.target_file = process_odb([odb, inp], state.cwd)

    elif user_input in hdf_options:
        # process hdf
        while True:
            user_input = input("Please enter the path of the hdf5 file, or the name of the hdf5 file in the hdfs directory: ")
            if(confirm(f"You entered {user_input}", "yes")):
                state.target_file = ensure_hdf(user_input, state.cwd)

            if state.target_file is not None:
                break

        state.target_file_name = state.target_file.split(".")[0]

    target_dir = os.path.dirname(state.target_file)
    target_file_config = os.path.join(target_dir, f"{state.target_file_name}.json")
    if os.path.exists(target_file_config):
        target_file_config_dict = state.load_json(target_file_config)
        state.set_mesh_seed_size(target_file_config_dict["seed"])
        print(f"Setting Default Seed Size of the Mesh to stored value of {state.mesh_seed_size}")
        state.meltpoint = target_file_config_dict["meltpoint"]
        state.select_colormap()
        print(f"Setting Default melpoint of the Mesh to stored value of {state.meltpoint}")

    else:
        set_seed_size(state)
        set_meltpoint(state)
        state.select_colormap()
        with open(target_file_config, "w") as tfc:
            json.dump({"seed": state.mesh_seed_size, "meltpoint": state.meltpoint}, tfc)


def get_extrema(state):
    while True:
        # Get the desired coordinates and time steps to plot
        extrema = {
                ("lower X", "upper X"): None,
                ("lower Y", "upper Y"): None,
                ("lower Z", "upper Z"): None,
                }
        for extremum in extrema.keys():
            extrema[extremum] = process_extrema(extremum)

        x_low, x_high = extrema[("lower X", "upper X")]
        y_low, y_high = extrema[("lower Y", "upper Y")]
        z_low, z_high = extrema[("lower Z", "upper Z")]
        print()
        if confirm(f"SELECTED VALUES:\nX from {x_low} to {x_high}\nY from {y_low} to {y_high}\nZ from {z_low} to {z_high}", "yes"):
            state.set_x_extrema(x_low, x_high)
            state.set_y_extrema(y_low, y_high)
            state.set_z_extrema(z_low, z_high)
            break


def set_seed_size(state):
    while True:
        try:
            step = float(input("Enter the Default Seed Size of the Mesh: "))

            if confirm(f"Default Seed Size: {step}", "yes"):
                state.set_mesh_seed_size(step)
                break

        except ValueError:
            print("Error, Default Seed Size must be a number")


def set_meltpoint(state):
    while True:
        try:
            meltpoint = float(input("Enter the meltpoint of the Mesh: "))

            if confirm(f"Meltpoint: {meltpoint}", "yes"):
                state.meltpoint = meltpoint
                break

        except ValueError:
            print("Error, melpoint must be a number")


def set_time(state):
    while True:
        values = [("lower time", 0, "0"), ("upper time", np.inf, "infinity")]
        lower_time = None
        upper_time = None
        for i, v in enumerate(values): 
            key, default, default_text = v
            while True:
                try:
                    user_input = input(f"Enter the {key} value you would like to plot (Leave blank for {default_text}): ")
                    if user_input == "":
                        result = default
                    else:
                        result = float(user_input)
                    
                    if i == 0:
                        lower_time = result
                    else:
                        upper_time = result
                    break

                except ValueError:
                    print("Error, all selected time values must be positive numbers")

        if confirm(f"You entered {lower_time} as the starting time and {upper_time} as the ending time.", "yes"):
            state.time_low = lower_time
            state.time_high = upper_time
            break


def process_extrema(keys):
    results = [None, None]
    for i, key in enumerate(keys):
        if i % 2 == 1:
            inf_addon = ""
            inf_val = np.inf
        else:
            inf_addon = "negative "
            inf_val = -1 * np.inf
        while True:
            try:
                user_input = input(f"Enter the {key} value you would like to plot (Leave blank for {inf_addon}infinity): ")
                if user_input == "":
                    results[i] = inf_val
                else:
                    results[i] = float(user_input)
                break

            except ValueError:
                print("Error, all selected coordinates and time steps must be numbers")

    return results


#  def process_input():
#      cwd = os.path.dirname(os.path.realpath(__file__))
#      input_files = ".hdf5 or (.odb and .inp)"
#
#      parser = argparse.ArgumentParser(description="ODB Extractor and Plotter")
#      parser.add_argument(input_files, nargs="*")
#
#      results = vars(parser.parse_args())[input_files]
#
#      if len(results) == 1:
#          # Ensure the hdf5 exists
#          target_file = ensure_hdf(results[0], cwd)
#      elif len(results) == 2:
#          # Process the given odb and inp files
#          target_file = process_odb(results, cwd)
#      else:
#          pass
#          #sys.exit("Error: You must supply either a .hdf5 file to read from or a pair of .odb and .inp files to process")
#
#      return cwd, target_file


def load_hdf(state):
    
    # First, ensure that the necessary parameters are set to load the file
    # Are x, y, and z set?
    if state.x.low is None or state.x.high is None or state.y.low is None or state.y.high is None or state.z.low is None or state.z.high is None:
        print('Error, you must set the physical extrema with the "extrema" or "range" commands')
        return

    if state.time_low is None or state.time_high is None:
        print('Error, you must set the time extrema with the "time" command')
        return

    if state.mesh_seed_size is None:
        print('Error, you must set the default seed size of the mesh with the "seed" "mesh" or "step" commands')
        return
    
    if state.target_file is None:
        print('Error, you must set the .hdf5 file you wish to open with the "select" command')
        return

    # Adapted from CJ's read_hdf5.py
    coords_df = get_coords(state.target_file)
    state.bounded_nodes = list(
            coords_df[
                (((coords_df["x"] == state.x.high) | (coords_df["x"] == state.x.low)) & ((coords_df["y"] >= state.y.low) & (coords_df["y"] <= state.y.high) & (coords_df["z"] >= state.z.low) & (coords_df["z"] <= state.z.high))) |
                (((coords_df["y"] == state.y.high) | (coords_df["y"] == state.y.low)) & ((coords_df["x"] >= state.x.low) & (coords_df["x"] <= state.x.high) & (coords_df["z"] >= state.z.low) & (coords_df["z"] <= state.z.high))) |
                (((coords_df["z"] == state.z.high) | (coords_df["z"] == state.z.low)) & ((coords_df["x"] >= state.x.low) & (coords_df["x"] <= state.x.high) & (coords_df["y"] >= state.y.low) & (coords_df["y"] <= state.y.high)))
                ]
                ["Node Labels"]
            )

    state.bounded_nodes_size = len(state.bounded_nodes)
    print(f"Extracting from {state.bounded_nodes_size} Nodes!")

    with Pool() as pool:
        # TODO can imap be used? starred imap?
        data = list()
        for node in state.bounded_nodes:
            data.append((node, state.target_file))
        results = pool.starmap(read_node_data, data)

    state.out_nodes = pd.concat(results)
    state.out_nodes = state.out_nodes[(state.out_nodes["Time"] <= state.time_high) & (state.out_nodes["Time"] >= state.time_low)]

    state.pre_process_data()


def get_views(state):
    while True:
        print("Please Select a Preset View for your plots: ")
        print('To view all default presets, please enter "list"')
        print ('Or, to specify your own view angle, please enter "custom"')
        print("Important Defaults: Top Face: 4, Right Face: 14, Front Face: 18, Top/Right/Front Isometric: 50")
        user_input = input("> ")
        if user_input.lower() == "list":
            print_views(state.views_list)
        elif user_input.lower() == "custom":
            elev, azim, roll = get_custom_views(state)
            state.angle = "Custom"
            state.elev = elev
            state.azim = azim
            state.roll = roll
            return
        else:
            try:
                user_input = int(user_input)
                if 0 > user_input > (len(state.views_list) + 1):
                    raise ValueError

                state.angle = state.views_list[user_input - 1]
                state.elev = state.views[state.views_list[user_input - 1]][0]
                state.azim = state.views[state.views_list[user_input - 1]][1]
                state.roll = state.views[state.views_list[user_input - 1]][2]
                return

            except ValueError:
                print(f'Error: input must be "list," "custom," or an integer between 1 and {len(state.views_list) + 1}')


def get_custom_views(state):
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

        if confirm(f"Elevation: {elev}\nAzimuth:   {azim}\nRoll:      {roll}", "yes"):
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
    print()


def plot_time_range(state):

    if not state.loaded:
        print('Error, you must load the contents of a .hdf5 file into memory with the "process" command in order to plot')
        return

    # out_nodes["Time"] has the time values for each node, we only need one
    # Divide length by len(bounded_nodes), go up to that
    times = state.out_nodes["Time"]
    final_time_idx = int(len(times) / state.bounded_nodes_size)
    # If you're showing every plot, do it slowly, in order
    if state.show_plots:
        for current_time in times[:final_time_idx]:
            plot_time_slice(current_time, times, state)

    # If each plot is not shown, batch-process, out of order.
    else:
        with Pool() as pool:
            print("Please wait while the plotter prepares your images...")
            # TODO can imap_unordered by used? starimap_unordered?
            data = list()
            for time in times[:final_time_idx]:
                data.append((time, times, state))
            pool.starmap(plot_time_slice, data)


def plot_time_slice(current_time, times, state):
    curr_nodes = state.out_nodes[times == current_time]

    current_time_name = format(round(current_time, 2), ".2f")
    file_name = state.target_file.split("/")[-1].split(".")[0]
    save_str = f"{state.results_dir}/{file_name}-{current_time_name}.png"
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = plt.axes(projection="3d", label=f"{file_name}-{current_time_name}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_box_aspect((state.x.size, state.y.size, state.z.size))
    ax.view_init(elev=state.elev, azim=state.azim, roll=state.roll)

    ax.set_title(f"3D Contour, time step: {current_time_name}")
    fig.add_axes(ax, label=f"{file_name}-{current_time_name}")

    if state.show_plots:
        print(f"Plotting time step {current_time_name}")

    faces = ["x_low", "x_high", "y_low", "y_high", "z_low", "z_high"]
    for face in faces:
        indices = ["X", "Y", "Z"]
        if "x" in face:
            indices.remove("X")
            y = state.y.vals
            z = state.z.vals

            if "low" in face:
                X = np.full((state.z.size, state.y.size), state.x.vals[0])
                temp_mask = curr_nodes["X"] == state.x.vals[0]
            elif "high" in face:
                X = np.full((state.z.size, state.y.size), state.x.vals[-1])
                temp_mask = curr_nodes["X"] == state.x.vals[-1]
            
            Y, Z = np.meshgrid(y, z)
            face_shape = X.shape

        elif "y" in face:
            indices.remove("Y")
            x = state.x.vals
            z = state.z.vals

            if "low" in face:
                Y = np.full((state.z.size, state.x.size), state.y.vals[0])
                temp_mask = curr_nodes["Y"] == state.y.vals[0]
            elif "high" in face:
                Y = np.full((state.z.size, state.x.size), state.y.vals[-1])
                temp_mask = curr_nodes["Y"] == state.y.vals[-1]
            
            X, Z = np.meshgrid(x, z)
            face_shape = Y.shape

        elif "z" in face:
            indices.remove("Z")
            x = state.x.vals
            y = state.y.vals

            if "low" in face:
                Z = np.full((state.y.size, state.x.size), state.z.vals[0])
                temp_mask = curr_nodes["Z"] == state.z.vals[0]
            elif "high" in face:
                Z = np.full((state.y.size, state.x.size), state.z.vals[-1])
                temp_mask = curr_nodes["Z"] == state.z.vals[-1]
            
            X, Y = np.meshgrid(x, y)
            face_shape = Z.shape

        temp_nodes = curr_nodes[temp_mask]
        first_dim = temp_nodes[indices[0]]  
        first_offset = first_dim.min()
        second_dim = temp_nodes[indices[1]]
        second_offset = second_dim.min()
        dim1, dim2 = face_shape
        colors = np.zeros((dim1, dim2, 3))
        for _, node in temp_nodes.iterrows():
            temp = node["Temp"]
            ind_2 = int((node[indices[0]] - first_offset) / state.mesh_seed_size)
            ind_1 = int((node[indices[1]] - second_offset) / state.mesh_seed_size)

            if temp >= state.meltpoint:
                colors[ind_1, ind_2] = (0.25, 0.25, 0.25)
            else:
                colors[ind_1, ind_2] = state.colormap.to_rgba(temp)[:3]

        ax.plot_surface(X, Y, Z, facecolors=colors)

    plt.savefig(save_str)
    if state.show_plots:
        plt.show()
    plt.close(fig)


#  def create_hollow_array(x, y, z):
#      x_ind, y_ind, z_ind = np.indices((x, y, z))
#      return ((x_ind <= 0 ) | (x_ind >= x - 1)) | ((y_ind <= 0) | (y_ind >= y - 1)) | ((z_ind <= 0 ) | (z_ind >= z - 1))


def ensure_hdf(input_file, cwd):
    if input_file == "":
        print("Error: .hdf5 file could not be found.")
        return
    cwd_file = os.path.join(cwd, input_file)
    hdfs_path = os.path.join(cwd, "hdfs")
    hdfs_path_file = os.path.join(hdfs_path, input_file)
    if not os.path.exists(cwd_file):
        if not os.path.exists(hdfs_path_file):
            print("Error: .hdf5 file could not be found.")
            return
        return hdfs_path_file
    return cwd_file


def process_odb(input_files, cwd):

    output_dir = os.path.join(cwd, "hdfs")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # First, odb_to_npz.py
    odb_file, inp_file = input_files

    filename = odb_file.split(".")[0]
    user_input = input(f"What would you like to name the output file (.hdf5 will be appended automatically). Enter nothing for the default {filename}.hdf5: ")

    if user_input != "":
        filename = user_input

    state.target_file_name = user_input
    output_file = f"{filename}.hdf5"

    odb_to_npz_args = ["abq2019", "python", "odb_to_npz.py", odb_file, inp_file]
    run(odb_to_npz_args)
    # By default, the odb_to_npz.py file creates tmp_npz, which we'll use as our goal file
    npz_dir = os.path.join(cwd, "tmp_npz")

    # Adapted from CJ's general purpose npz to hdf code
    # Convert to HD5
    os.chdir(output_dir)
    with h5py.File(output_file, "w") as hdf5_file:
        for root, _, files in os.walk(npz_dir, topdown=True):
            for filename in files:
                item = os.path.join(root, filename)
                read_npz_to_hdf(item, npz_dir, hdf5_file)

    os.chdir(cwd)
    if os.path.exists(npz_dir):
        shutil.rmtree(npz_dir)

    return os.path.join(output_dir, output_file)


def read_npz_to_hdf(item, npz_dir, hdf5_file):
    npz = np.load(item)
    arr = npz[npz.files[0]]
    item_name = os.path.splitext(item)[0].replace(npz_dir, "")
    hdf5_file.create_dataset(item_name, data=arr, compression="gzip")


def get_coords(hdf5_filename):
    """Gets all coordinates of the HDF5 file related to its nodes."""
    with h5py.File(hdf5_filename, "r") as hdf5_file:
       coords = hdf5_file["node_coords"][:]
       node_labels, x, y, z = np.transpose(coords)
    out_data = pd.DataFrame.from_dict({"Node Labels": node_labels.astype(int), "x": x, "y": y, "z": z})
    return out_data


def read_node_data(node_label, hdf5_filename):
    """Creates a long format DataFrame with rows being nodes that represent different important information per node."""
    with h5py.File(hdf5_filename, "r") as hdf5_file:
        coords = hdf5_file["node_coords"][:]
        node_coords = coords[np.where(coords[:, 0] == node_label)[0][0]][1:]

        sys.exit(print(hdf5_file))

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
