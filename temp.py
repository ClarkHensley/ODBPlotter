
import h5py

# At time T1

# We need to associate T1 with the given Frame
# But if Frame is known as F1

# X, Y, Z --> X1, Y1, Z1

# data[data["X"] == X1 && data["Y"] == X2 && data["Z"] == Z1 && data["Frame"] == F1]

with h5py.File("hdfs/test_stepped.hdf5", "r") as hdf5_file:
    for key in hdf5_file:
        print(key)
        print(hdf5_file[key])
