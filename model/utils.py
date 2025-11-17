import numpy as np
import os
import astropy.constants as c
import pandas as pd

results_folder = '/Users/ah258874/PycharmProjects/punto/results_model'


def find_nearest(value, array):
    print('TBD')
    wave_value = value
    ind_value = np.where(wave_value)

    return wave_value, ind_value


def convert_deg_to_radian(x):
    return x * (np.pi/180)


def save_model(system):
    np.savetxt(os.path.join(results_folder, f'photometrymodel_{(system.planetaryradius / c.R_jup).value}'
                                            f'Rj_Ab{system.albedo}_r_{system.redistribution}.txt'),
               np.transpose(np.array([system.phase * 180 / np.pi, system.contrast_ppm])))


def compute_edges(arr):
    """Compute cell edges from center values."""
    midpoints = 0.5 * (arr[1:] + arr[:-1])
    first = arr[0] - (midpoints[0] - arr[0])
    last = arr[-1] + (arr[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def phase_fold(x, y, period, t0):
    df = pd.DataFrame(np.column_stack((x, y)), columns=['t', 'f'])

    # t0 = df['t'][0]
    df['p'] = (df['t'] - t0) % period - 0.5 * period

    df = df.sort_values(by='p').reset_index(drop=True)

    df = df.groupby(df['p'].index).mean()

    return df['p'], df['f']


def bin_data(array, bin_size, err=None):
    # Gets the remainder of the floor division between lightcurve size and bin size
    division_remainder = np.mod(len(array), bin_size)

    if err is not None:
        tmp_err = err[division_remainder:]

    # We  remove the points that could  not be  part of a full bin
    tmp_data = array[division_remainder:]

    binned_array = []
    binned_err = []
    length = int(len(tmp_data) / bin_size)

    # We bin the data
    for i in range(length):
        tmp_bin = np.mean(tmp_data[(i * bin_size):((i + 1) * bin_size)])
        binned_array.append(tmp_bin)
        if err is not None:
            tmp_binned_err = np.sqrt(np.sum(tmp_err[(i * bin_size):((i + 1) * bin_size)] ** 2) / (bin_size**2))
            binned_err.append(tmp_binned_err)

    return np.asarray(binned_array), np.asarray(binned_err)


def compute_binning_time(time_array, bin_size):
    bin_time = time_array[bin_size] - time_array[0]

    return bin_time  # day