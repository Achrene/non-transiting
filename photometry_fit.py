import numpy as np
import matplotlib.pyplot as plt
import ultranest.stepsampler

import corner
import os
import pandas as pd
import cmocean as cm

from matplotlib import rc
rc('image', origin='lower')
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 16})
rc('text', usetex=True)
rc('lines', linewidth=0.5)
rc('ytick', right=True, direction = 'in')
rc('xtick', top = True, direction = 'in')
rc('axes', axisbelow = False)
rc('mathtext', fontset = 'cm')

colors_matter = cm.cm.matter_r(np.linspace(0,1,10))


def phase_fold(x, y, period, t0):
    df = pd.DataFrame(np.column_stack((x, y)), columns=['t', 'f'])

    # t0 = df['t'][0]
    df['p'] = (df['t'] - t0) % period - 0.5 * period

    df = df.sort_values(by='p').reset_index(drop=True)

    df = df.groupby(df['p'].index).mean()

    return df['p'], df['f']


def transform_uniform(cube, a, b):
    params = cube.copy()
    # transform location parameter: uniform prior
    params = cube * (b - a) + a

    return params


def binning(lightcurve, bin_size):
    # Gets the remainder of the floor division between lightcurve size and bin size
    division_remainder = np.mod(len(lightcurve), bin_size)

    # We  remove the points that could  not be  part of a full bin
    tmp_data = lightcurve[division_remainder:]

    binned_lightcurve = []
    length = int(len(tmp_data) / bin_size)

    # We bin the data
    for i in range(length):
        tmp_bin = np.mean(tmp_data[(i * bin_size):((i + 1) * bin_size)])
        binned_lightcurve.append(tmp_bin)

    return np.asarray(binned_lightcurve)


if __name__ == '__main__':

    run = True

    period = 0.604734  # in days

    input_file = '/Users/ah258874/Documents/STScI_fellowship/Punto/Kepler_long_cadence/9139163_lc_filtered.txt'
    general_folder = '/Users/ah258874/Documents/STScI_fellowship/Punto/fits/photometric_fit'
    out_folder = os.path.join(general_folder, 'test')

    data = np.loadtxt(input_file)
    time = data[:, 0]  # binning(data[:, 0], 10)
    y = data[:, 1]  # binning(data[:, 1], 10)  #
    ref_time = data[0, 0] - 0.27 * period

    yerr = 1

    foldx, foldy = phase_fold(time, y, period, ref_time)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
    ax.errorbar(foldx, foldy, yerr=yerr, color='k', zorder=2, alpha=0.2, label='data')
    ax.set_xlabel('Time [day]')
    ax.set_ylabel('Amplitude [ppm]');
    plt.tight_layout()
    plt.legend()
    plt.show()

    def model(time, A0, Arefl, T0, Aellip, Abeam):
        phi = (time - T0) / period
        return A0 - Arefl * np.cos(2 * np.pi * phi) - Aellip * np.cos(4 * np.pi * phi) \
               + Abeam * np.sin(2 * np.pi * phi)

    def my_likelihood(params):
        A0, Arefl, T0, Aellip, Abeam = params

        # model
        y_model = model(time, A0, Arefl, T0, Aellip, Abeam)

        # compare model and data with gaussian likelihood:
        like = -0.5 * (((y_model - y) / yerr) ** 2).sum()

        return like


    priors = [(-1, 1)] + [(-100.0, 100.0)] + [(-0.2, 0)] + [(-100.0, 100.0)] + [(-100.0, 100.0)]
    ndim = len(priors)

    def prior(cube):  # ndim, params
        params = cube.copy()

        for i in range(ndim):
            params[i] = transform_uniform(cube[i], priors[i][0], priors[i][1])

        return params


    param_names = ['A0', 'Arefl', 'T0', 'Aellip', 'Abeam']
    if run:
        sampler = ultranest.ReactiveNestedSampler(param_names, my_likelihood, prior, log_dir=out_folder,
                                                  resume='overwrite')
        output = sampler.run()
        sampler.print_results()

    else:
        output = ultranest.integrator.read_file(out_folder, ndim)[-1]

    flat_samples = output['samples']
    num_samples = flat_samples.shape[0]

    # Corner plot
    fig = corner.corner(flat_samples, labels=param_names, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt='.5f', title_kwargs={"fontsize": 12})
    fig.savefig(os.path.join(out_folder + '/plots', 'corner.pdf'), format='pdf')
    plt.show()

    fit_stat_values = np.zeros((ndim, 3))
    for i in range(ndim):
        fit_stat = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(fit_stat)
        fit_stat_values[i, 0] = fit_stat[1]
        fit_stat_values[i, 1] = -q[0]
        fit_stat_values[i, 2] = q[1]

    txt_array = pd.DataFrame(data={'param': pd.Series(param_names, dtype="string"), 'mean': fit_stat_values[:, 0],
                                   'error-': fit_stat_values[:, 1], 'error+': fit_stat_values[:, 2]})

    np.savetxt(os.path.join(out_folder + '/results_model', 'params.txt'), txt_array, fmt='%s')

    foldx_fit, foldy_fit = phase_fold(time, model(time, fit_stat_values[0, 0],
                                                  fit_stat_values[1, 0],
                                                  fit_stat_values[2, 0],
                                                  fit_stat_values[3, 0],
                                                  fit_stat_values[4, 0]), period, fit_stat_values[2, 0])
    foldx, foldy = phase_fold(time, y, period, fit_stat_values[2, 0])

    # Plot the fit
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=100)
    ax.plot(foldx_fit, foldy_fit, color = colors_matter[3], linewidth=2)
    ax.errorbar(foldx, foldy, yerr=yerr, color='k', zorder=2, alpha=0.2, label='data')
    ax.set_xlabel('Time [day]')
    ax.set_ylabel('Amplitude [ppm]');
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(out_folder + '/plots', 'best_fit.pdf'), format='pdf')
    plt.show()

    print('done')