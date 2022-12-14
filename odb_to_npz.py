# Author: CJ Nguyen
# Heavily based off of example ODB extraction script from the CMML Github written by Will Furr and Matthew Dantin
# This script takes an ODB and converts the nodeset data to npz files
# The data that is created is the following
# * npzs containing temperatures for each node at every frame, one dataframe per step of the program
# * npz containing coordinate data for each node, organized by nodelabel - xyz coordinate
# * npz containing the starting time of each frame

# Usage: python <script name> <odb file name> <inp file name>
# Where <inp file name> is the .inp file used to generate the ODB (needed for the timing of each frame)
# NOTE: This script makes three major assumptions:
# * The Odb has a part named "PART-1-1"
# * The part's first nodeset is the nodeset that references all nodes
# * The first frame of the sequence outputs the coordinates of all nodes
# Without these assumptions, the script will have unexpected behavior.

# This script can be configured by having a file of name `odb_to_npz_config.json` in the working directory.
# The format of this file is specified in the readme.

import sys
import os
from odbAccess import *
from abaqusConstants import *
from types import IntType
import numpy as np
import threading
import multiprocessing
import json

# Constants
MAX_THREADS = 50
MAX_WORKERS = 8
CONFIG_FILENAME = "odb_to_npz_config.json"
default_config = {
    "max_processors": 8,
    "nodeset_max_threads": 50,
    "step_temps_max_threads": 20
}

# Create directory to store npzs
out_dir = os.path.join(os.getcwd(), "tmp_npz")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
coord_file = os.path.join(out_dir, "node_coords.npz")
frame_time_file = os.path.join(out_dir, "step_frame_increments.npz")

def read_nodeset_items(sema, nodesets, outdir, nodeset_name):
    # Multithreading intended function for reading nodeset data
    sema.acquire()
    print("\tExtracting nodeset data from nodeset {}".format(nodeset_name))
    out_nodeset_name = os.path.join(outdir, nodeset_name)
    out_nodeset_name += ".npz"
    np.savez_compressed(out_nodeset_name, np.array([node.label for node in nodesets[nodeset_name].nodes]))
    print("\tFinished processing nodeset {}".format(nodeset_name))
    sema.release()


def read_step_data(odb_filename, step_name):
    # Multithreading intended function for reading steps' data
    frame_threading_sema = threading.Semaphore(config["step_temps_max_threads"])
    
    # Error if I don't repeat these steps
    odb = openOdb(odb_filename, readOnly=True)
    steps = odb.steps
    assembly = odb.rootAssembly

    print("Working on temperatures from step: {}".format(step_name))
    curr_step_dir = os.path.join(temps_dir, step_name)
    if not os.path.exists(curr_step_dir):
        os.mkdir(curr_step_dir)

    step_frame_extraction_threads = list()
    for i in range(len(steps[step_name].frames)):
        thread = threading.Thread(target=read_frame_temp, args=(frame_threading_sema, steps, assembly, step_name, i, curr_step_dir))
        thread.start()
        step_frame_extraction_threads.append(thread)
    for thread in step_frame_extraction_threads:
        thread.join()
    odb.close()

def unpack_read_step_data(args):
    read_step_data(*args)


def read_frame_temp(sema, steps, assembly, step_name, frame_num, outdir):
    # Multithreading function for reading information from frames
    sema.acquire()
    frame = steps[step_name].frames[frame_num]
    field = frame.fieldOutputs['NT11'].getSubset(region=assembly.instances['PART-1-1'].nodeSets[assembly.instances['PART-1-1'].nodeSets.keys()[0]])
    node_temps = []
    for item in field.values:
        # e.g. for node in values
        node = item.nodeLabel    # unnecessary
        temp = item.data
        node_temps.append(temp)
    np.savez_compressed(os.path.join(outdir, "frame_{}".format(frame_num)), np.array(node_temps))
    sema.release()

def read_frame_coords(frame):
    # The below reference pulls from the nodeset representing all nodes
    coords = frame.fieldOutputs['COORD'].getSubset(region=assembly.instances['PART-1-1'].nodeSets[assembly.instances['PART-1-1'].nodeSets.keys()[0]])
    print("\tGetting node coordinates")

    coord_arr = []
    for item in coords.values:
        # e.g. for node in values
        node = item.nodeLabel
        coord = item.data    # outputs [xcoord, ycoord, zcoord]
        xyz = []
        for axis in coord:
            xyz.append(axis)
        coord_arr.append([node, xyz[0], xyz[1], xyz[2]])
    np.savez_compressed(coord_file, np.array(coord_arr))

    pass

if __name__ == "__main__":
    args = sys.argv

    odb_filename = args[1]
    inp_filename = args[2]

    # Read config if it exists or use default config
    if os.path.isfile(CONFIG_FILENAME):
        with open(CONFIG_FILENAME, 'r') as config_file:
            config = json.load(config_file)
    else:
        config = default_config

    # Create multithread limiter object

    # FRAME TIME SECTION
    # --------------------

    with open(inp_filename, 'r') as file:
        time_dir = os.path.join(out_dir, "step_frame_times")
        if not os.path.exists(time_dir):
            os.mkdir(time_dir)
        last_total = 0
        inp_file = file.readlines()
        for i in range(len(inp_file)):
            if "*STEP" in inp_file[i]:
                step_line = inp_file[i]
                name = re.search("(?<=, )NAME=.+?(?=,)", step_line).group(0).split('=')[1]
                increment_line = inp_file[i + 2]
                increment = float(increment_line.split(", ")[0])
                total_time = float(increment_line.split(", ")[1])
                times = list(np.arange(last_total, last_total + total_time + increment, increment))
                last_total += total_time + increment
                np.savez_compressed("{}.npz".format(os.path.join(time_dir, name)), times)

    # ODB ACCESS SECTION
    # --------------------

    odb = openOdb(odb_filename, readOnly=True)
    assembly = odb.rootAssembly
    steps = odb.steps

    # EXTRACT NODESET DATA
    # --------------------

    # Create output directory for nodesets
    nodeset_dir = os.path.join(out_dir, "nodesets")
    if not os.path.exists(nodeset_dir):
        os.mkdir(nodeset_dir)
    # Loop through all the nodesets and get the nodes that they cover
    nodesets = assembly.instances['PART-1-1'].nodeSets # returns a dictionary of ODB objects

    nodeset_thread_sema = threading.Semaphore(config["nodeset_max_threads"])
    nodeset_extraction_threads = list()
    for key in nodesets.keys():
        thread = threading.Thread(target=read_nodeset_items, args=(nodeset_thread_sema, nodesets, nodeset_dir, key))
        thread.start()
        nodeset_extraction_threads.append(thread)
    # Extract coordinates
    coord_thread = threading.Thread(target=read_frame_coords, args=(steps[steps.keys()[0]].frames[0],))
    coord_thread.start()
    nodeset_extraction_threads.append(coord_thread)
    for thread in nodeset_extraction_threads:
        thread.join()

    # EXTRACT NODESET DATA
    # --------------------

    # Create output directory for temperatures
    temps_dir = os.path.join(out_dir, "temps")
    if not os.path.exists(temps_dir):
        os.mkdir(temps_dir)

    # Get argument tuples for extraction function and close ODB
    step_temps_args = [(odb_filename, step_name) for step_name in steps.keys()]
    odb.close()

    # Begin processes that will get temperature data for each frame of each step
    pool = multiprocessing.Pool(config["max_processors"])
    pool.map(unpack_read_step_data, step_temps_args)
    pool.close()
    # step_extraction_threads = list()
    # for step in steps.keys():
    #     thread = threading.Thread(target=read_step_data, args=(, steps, step))
    #     thread.start()
    #     step_extraction_threads.append(thread)

    # for thread in step_extraction_threads:
    #     thread.join()
