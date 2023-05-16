import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def read_montage(fname):
    x = []
    y = []
    z = []
    with open(fname, "r") as f:
        for line in f:
            parts = line.split("\t")
            if parts[0][0] == "E":
                x.append(float(parts[1]))
                y.append(float(parts[2]))
                z.append(float(parts[3]))

    x = np.array([x, y, z])
    return x

def cartesian_to_spherical(montage_xyz):
    montage_spher = np.zeros(montage_xyz.shape)
    xy = np.sum(np.power(montage_xyz[0:2], 2), axis=0)
    montage_spher[0] = np.sqrt(xy + np.power(montage_xyz[2], 2))              # r
    montage_spher[1] = np.arctan2(np.sqrt(xy), montage_xyz[2])      # theta
    montage_spher[2] = np.arctan2(montage_xyz[1], montage_xyz[0])   # phi

    return montage_spher

def spherical_to_cartesian(montage_spher):
    montage_approx = np.zeros((2, montage_spher.shape[1]))
    montage_approx[0] = montage_spher[1] * np.cos(montage_spher[2])
    montage_approx[1] = montage_spher[1] * np.sin(montage_spher[2])
    return montage_approx

def montage_to_2d(montage):
    montage_spher = cartesian_to_spherical(montage)
    return spherical_to_cartesian(montage_spher)

def transform_montage(montage, nx, ny):
    montage2d = montage_to_2d(montage)
    grid_size = np.array([nx, ny])
    montage2d_bin = (montage2d.T - np.amin(montage2d, axis=1)).T
    montage2d_bin = np.round((montage2d_bin.T / np.amax(montage2d_bin, axis=1) * grid_size).T).astype(int)

    return montage2d_bin

def transform_2d_to_3d(x2d, montage2d_bin, nx, ny):
    assert(montage2d_bin.shape[1] == x2d.shape[0]) # Number of channels match
    x3d = np.zeros((nx+1, ny+1, x2d.shape[1]))

    for i in range(nx+1):
        for j in range(ny+1):
            ind_x = montage2d_bin[0] == i
            ind_y = montage2d_bin[1] == j
            equal = np.logical_and(ind_x, ind_y)
            if np.sum(equal) > 1:
                raise Exception(f"Error: several electrodes at (x,y)= ({i}, {j})")

            if np.any(equal):
                x3d[i,j] = x2d[np.nonzero(equal)[0]]

    return x3d


def crop_montage(montage2d_bin, x_l, x_h, y_l, y_h):
    montage_cropped = (montage2d_bin[x_l:x_h])[:,y_l:y_h]
    return montage_cropped