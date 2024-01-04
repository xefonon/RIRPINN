import sys

sys.path.append('../')
import torch
import torch.autograd as autograd  # computation graph
import torch.nn as nn  # neural networks
import numpy as np
from torch.utils.data import Dataset
from src.utils_soundfields import plot_sf, plot_sf_quiver
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob
import re
from pyDOE import lhs
from src.SIREN import SirenNet, Siren, SirenResnet
import h5py
from sklearn.neighbors import NearestNeighbors
import librosa
import json
import yaml


class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def dict2obj(d):
    return json.loads(json.dumps(d), object_hook=obj)


def config_from_yaml(yamlFilePath, no_description=True):
    def changer(config):
        for attr, value in vars(config).items():
            try:
                setattr(config, attr, value.value)
            except:
                ValueError("problem with config: {} and value: {}".format(config, value))

    with open(yamlFilePath) as f:
        # use safe_load instead load
        dataMap = yaml.safe_load(f)
    config = dict2obj(dataMap)
    if no_description:
        changer(config)
    return config


def spatial_nyquist(grid, c=343.):
    """
    Parameters
    ----------
    grid - uniform grid (e.g. square or cube) in cartesian coords [3 x Npoints]
    c - speed of sound

    Returns
    -------
    fnyq - nyquist frequency for each dimension [3,]
    """

    xcoord = np.sort(grid[0])
    ycoord = np.sort(grid[1])
    zcoord = np.sort(grid[2])

    dx = np.diff(xcoord)
    dx = dx[dx.nonzero()]
    dy = np.diff(ycoord)
    dy = dy[dy.nonzero()]
    dz = np.diff(zcoord)
    dz = dz[dz.nonzero()]

    fnyq_x = c / (2 * dx)
    fnyq_y = c / (2 * dy)
    fnyq_z = c / (2 * dz)
    return fnyq_x, fnyq_y, fnyq_z, [dx, dy, dz]


# # get spatial grid in xy plane which is 4 m x 4 m
# x = np.linspace(-2, 2, 30)
# y = np.linspace(-2, 2, 30)
# X, Y = np.meshgrid(x, y)
# z = np.zeros_like(X)
# grid = np.vstack((X.reshape(1, -1), Y.reshape(1, -1), z.reshape(1, -1)))

def subsample_gridpoints(grid, subsample=None, comparison_grid=None):
    r0 = grid.mean(axis=-1)
    tempgrid = grid - r0[:, None]
    xmin, xmax = round(tempgrid[0].min(), 3), round(tempgrid[0].max(), 3)
    # ymin, ymax = round(tempgrid[1].min(), 3), round(tempgrid[1].max(), 3)
    if subsample is None:
        subsample = grid.shape[-1]
    if subsample == grid.shape[-1] and comparison_grid is None:
        indices = np.arange(grid.shape[-1]).reshape(-1, 1)
        return grid[:, indices.squeeze(-1)], indices.squeeze(-1)
    elif comparison_grid is not None:
        newgrid = comparison_grid
    else:
        newgrid = reference_grid(subsample, xmin, xmax)
        newgrid += r0[:2, None]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(grid[:2].T)
    distances, indices = nbrs.kneighbors(newgrid.T)
    return grid[:, indices.squeeze(-1)], indices.squeeze(-1)


def reference_grid(steps, xmin=-.7, xmax=.7):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(xmin, xmax, steps)
    # z = tf.zeros(shape = (steps,))
    X, Y = np.meshgrid(x, y)
    gridnew = np.vstack((X.reshape(1, -1), Y.reshape(1, -1)))
    return gridnew


def plot_frequency_response(H, fs=8000, ax=None, normalize=True,
                            cnvrt_to_freq=True, color='k', **kwargs):
    if cnvrt_to_freq:
        f = np.fft.rfftfreq(H.shape[-1], 1 / fs)
        H = np.fft.rfft(H, axis=-1)
    normalization = lambda x: x / np.max(np.abs(x))
    if H.ndim < 2:
        H = H[None, :]
    if len(H) > 1:
        labels = ['Reference', 'Prediction']
    else:
        labels = [None]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, h in enumerate(H):
        if normalize:
            h = normalization(h)
        ax.plot(f, 20 * np.log10(np.abs(h)), color=color, label=labels[i])
    ax.set_xlabel('frequency (Hz)')
    # ax.set_ylabel('Magnitude [dB]')
    ax.grid(which='both', ls=':', color='k')
    ax.legend()
    return ax


def speed_of_sound(T):
    """
    speed_of_sound(T)
    Caculate the adiabatic speed of sound according to the temperature.
    Parameters
    ----------
    T : double value of temperature in [C].
    Returns
    -------
    c : double value of speed of sound in [m/s].
    """
    c = 20.05 * np.sqrt(273.15 + T)
    return c


def load_measurement_data(filename):
    with h5py.File(filename, "r") as f:
        data_keys = f.keys()
        meta_data_keys = f.attrs.keys()
        data_dict = {}
        for key in data_keys:
            try:
                data_dict[key] = f[key][:]
            except:
                data_dict[key] = f[key][()]
        for key in meta_data_keys:
            data_dict[key] = f.attrs[key]
        f.close()
    return data_dict


class standardize_rirs:
    def __init__(self, data, device='cuda'):
        self.data = data
        self.tfnp = lambda x: torch.from_numpy(x).float().to(device)

        self.mean = self.tfnp(data.mean()[None, None])
        self.std = self.tfnp(data.std()[None, None])

    def forward_rir(self, input):
        return (input - self.mean) / self.std

    def backward_rir(self, input):
        return input * self.std + self.mean


# class standardize_rirs:
#     def __init__(self, data, device='cuda'):
#         self.data = data
#         self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
#
#         self.mean = self.tfnp(data.mean(axis=0)[None, :])
#         self.std = self.tfnp(data.std(axis=0)[None, :])
#
#     def forward_rir(self, input):
#         return (input - self.mean) / self.std
#
#     def backward_rir(self, input):
#         return input * self.std + self.mean

class unit_norm_normalization:
    def __init__(self, data, device='cuda'):
        self.data = data
        self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
        self.l2_norm = lambda x: np.linalg.norm(x)

        self.norm = self.tfnp(self.l2_norm(data)[None, None])

    def forward_rir(self, input):
        return input / self.norm

    def backward_rir(self, input):
        return input * self.norm


class normalize_rirs:
    def __init__(self, data, device='cuda'):
        self.data = data
        self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
        self.lb = data.min()
        self.ub = data.max()
        self.scaling = lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        self.unscaling = lambda x: (x + 1) * (self.ub - self.lb) / 2. + self.lb

        # self.norm = self.tfnp(self.maxabs(data)[None, None]/.95)

    def forward_rir(self, input):
        return self.scaling(input)

    def backward_rir(self, input):
        return self.unscaling(input)

    def forward_rir(self, input):
        return input / self.norm

    def backward_rir(self, input):
        return input * self.norm


class normalize_rirs:
    def __init__(self, data, device='cuda'):
        self.data = data
        self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
        self.lb = data.min()
        self.ub = data.max()
        self.scaling = lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        self.unscaling = lambda x: (x + 1) * (self.ub - self.lb) / 2. + self.lb

        # self.norm = self.tfnp(self.maxabs(data)[None, None]/.95)

    def forward_rir(self, input):
        return self.scaling(input)

    def backward_rir(self, input):
        return self.unscaling(input)


class maxabs_normalize_rirs:
    def __init__(self, data, device='cuda', l_inf_norm=0.1):
        self.data = data
        self.tfnp = lambda x: torch.from_numpy(x).float().to(device)
        self.maxabs = lambda x: np.max(abs(x))
        self.l_inf_norm = l_inf_norm

        self.norm = self.tfnp(self.maxabs(data)[None, None])

    def forward_rir(self, input):
        return self.l_inf_norm * input / (self.norm)

    def backward_rir(self, input):
        return (1 / self.l_inf_norm) * input * self.norm


def get_odeon_data(filename, subsample_points=10):
    if subsample_points in ['None', None, 'none']:
        subsample_points = None
    data_dict = load_measurement_data(filename)
    rirs = data_dict['rirs']
    receiver_grid = data_dict['receiver_grid']
    # if receiver grid first dim is 3 then transpose
    if receiver_grid.shape[-1] == 3:
        receiver_grid = receiver_grid.T
    # center receiver grid
    receiver_grid[0] = receiver_grid[0] - np.mean(receiver_grid[0])
    receiver_grid[1] = receiver_grid[1] - np.mean(receiver_grid[1])
    source_grid = data_dict['source_grid']
    fs = data_dict['fs']
    temperature = data_dict['temperature']
    c = speed_of_sound(temperature)
    # training data
    refdata = librosa.resample(y=rirs, orig_sr=fs, target_sr=8000)
    fs = 8000
    grid_measured, indcs = subsample_gridpoints(receiver_grid, subsample=subsample_points)
    measureddata = refdata[indcs]
    return refdata, fs, receiver_grid, measureddata, grid_measured, c


def get_jabra_data(filename):
    """ This function loads the jabra data (h5 format) from a given filename."""
    data_dict = load_measurement_data(filename)
    measureddata = data_dict['rirs']
    initial_pressure_data = data_dict['ref_rirs']
    fs = data_dict['fs']
    initial_pressure_data = librosa.resample(initial_pressure_data, orig_sr=fs, target_sr=8000)
    measureddata = librosa.resample(measureddata, orig_sr=fs, target_sr=8000)
    fs = 8000
    grid_measured = data_dict['mic_grid'].T
    grid_initial = data_dict['ref_mic_grid'].T
    loudspeaker_position = data_dict['loudspeaker_position']
    c = speed_of_sound(19.5)
    # find onset index in initial pressure data
    max_indices = [np.argmax(abs(initial_pressure_data[i, :])) for i in range(initial_pressure_data.shape[0])]
    onset_index = np.min(max_indices) - int(.05 * np.min(max_indices))
    measureddata = measureddata[:, onset_index:]
    # create meshgrid for x = -0.5 to 2. and y = -0.15 to 2.3
    x = np.linspace(-0.5, 2., 100)
    y = np.linspace(-0.15, 2.3, 100)
    X, Y = np.meshgrid(x, y)
    gridref = np.vstack((X.flatten(), Y.flatten()))
    # plot grids
    # plt.figure()
    # plt.plot(gridref[0, :], gridref[1, :], '.')
    # plt.plot(grid_measured[:, 0], grid_measured[:, 1], 'o')
    # plt.plot(grid_initial[:, 0], grid_initial[:, 1], 'o')
    # plt.plot(loudspeaker_position[0], loudspeaker_position[1], 'o')
    # plt.show()

    initial_pressure_data = initial_pressure_data[:, onset_index:]
    return (measureddata, initial_pressure_data, fs, grid_measured, grid_initial, gridref,
            loudspeaker_position, c)


def get_019_measurement_vectors(filename, subsample_points=10):
    data_dict = load_measurement_data(filename)
    refdata = data_dict['rirs']
    temperature = 18.5
    c = speed_of_sound(temperature)
    fs = data_dict['fs']
    refdata = librosa.resample(refdata, fs, 8000)
    fs = 8000
    grid = data_dict['grid'].T / 1000
    measureddata = refdata
    # grid_measured = data_dict['grid_bottom']
    # grid -= grid.mean(axis=-1)[:, None]
    grid_measured, indcs = subsample_gridpoints(grid, subsample=subsample_points)
    measureddata = measureddata[indcs]
    return refdata, fs, grid, measureddata, grid_measured, c


def add_noise(data, snr):
    """ This function adds noise to the given data with a given signal to noise ratio."""
    # calculate noise power
    noise_power = np.mean(data ** 2) / (10 ** (snr / 10))
    # generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    # add noise to data
    data = data + noise
    return data


def get_measurement_vectors(filename, subsample_points=10):
    if subsample_points in ['None', None, 'none']:
        subsample_points = None
    data_dict = load_measurement_data(filename)
    refdata = data_dict['RIRs_bottom']
    temperature = data_dict['temperature']
    c = speed_of_sound(temperature)
    fs = data_dict['fs']
    refdata = librosa.resample(y=refdata, orig_sr=fs, target_sr=8000)
    fs = 8000
    grid = data_dict['grid_bottom']
    measureddata = refdata
    # grid_measured = data_dict['grid_bottom']
    grid -= grid.mean(axis=-1)[:, None]
    grid_measured, indcs = subsample_gridpoints(grid, subsample=subsample_points)
    measureddata = measureddata[indcs]
    return refdata, fs, grid, measureddata, grid_measured, c


def get_simulated_measurement_vectors(filename, subsample_points=10, additive_noise=True, snr=20):
    if subsample_points in ['None', None, 'none']:
        subsample_points = None
    data_dict = load_measurement_data(filename)
    if ('ism' or 'ISM') in filename:
        ism = True
    if not ism:
        refdata = data_dict['rirs']
        temperature = data_dict['temperature']
        c = speed_of_sound(temperature)
        fs = data_dict['fs']
        refdata = librosa.resample(y=refdata, orig_sr=fs, target_sr=8000)
        grid = data_dict['grid_mic']
        fs = 8000
    else:
        refdata = data_dict['rirs']
        c = 343.
        fs = data_dict['fs']
        grid = data_dict['grid']
    if additive_noise:
        refdata = add_noise(refdata, snr)
    measureddata = refdata
    # grid_measured = data_dict['grid_bottom']
    grid -= grid.mean(axis=-1)[:, None]
    grid_measured, indcs = subsample_gridpoints(grid, subsample=subsample_points)
    measureddata = measureddata[indcs]
    return refdata, fs, grid, measureddata, grid_measured, c


def save_checkpoint(directory, filepath, obj, remove_below_step=None, current_step=None):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")
    if remove_below_step is not None:
        print("\nRemoving checkpoints below step ", remove_below_step)
        remove_checkpoint(directory, remove_below_step)
        # keep only one checkpoint per 10000 steps below current 10000
        # find nearest multiple of 10000 below current step
        multiple = 10000 * (current_step // 10000)
        if current_step is not None:
            remove_checkpoint_multiple(directory, multiple=multiple, delete_below_steps=current_step - 10000)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def remove_checkpoint(cp_dir, delete_below_steps=1000):
    filelist = [f for f in os.listdir(cp_dir) if f.startswith("PINN")]
    for f in filelist:
        prefix, number, extension = re.split(r'(\d+)', f)
        if int(number) < delete_below_steps:
            os.remove(os.path.join(cp_dir, f))


def remove_checkpoint_multiple(cp_dir, multiple=10000, delete_below_steps=1000):
    filelist = [f for f in os.listdir(cp_dir) if f.startswith("PINN")]
    for f in filelist:
        prefix, number, extension = re.split(r'(\d+)', f)
        if int(number) < delete_below_steps and int(number) % multiple == 0:
            os.remove(os.path.join(cp_dir, f))


def construct_input_vec(x_true, y_true, t, rirdata=None, data_ind=None, t_ind=None):
    if data_ind is not None:
        if rirdata is not None:
            rirdata = rirdata[data_ind]
        x_true = x_true[data_ind]
        y_true = y_true[data_ind]
    if t_ind is not None:
        if rirdata is not None:
            rirdata = rirdata[:, t_ind]
        t = t[t_ind]
    collocation = []
    for i in range(len(t)):
        tt = np.repeat(t[i], len(x_true))
        collocation.append(np.stack([x_true, y_true, tt], axis=0))
    return np.array(collocation), rirdata


def construct_rir_input_vec(x_true, y_true, t, t_ind=None):
    if t_ind is not None:
        t = t[t_ind]
    xx = np.repeat(x_true, len(t))
    yy = np.repeat(y_true, len(t))
    collocation = np.stack([xx, yy, t], axis=0)
    return np.array(collocation)


def plot_without_errors(bounds, PINN, grids=None):
    tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)
    x_col = np.linspace(bounds['x'][0], bounds['x'][1], 100)
    y_col = np.linspace(bounds['y'][0], bounds['y'][1], 100)
    t_col = np.linspace(.0001, 0.03, 15)
    x_mesh, y_mesh, t_mesh = np.meshgrid(x_col, y_col, t_col)
    collocation_data = np.stack((x_mesh.flatten(), y_mesh.flatten(), t_mesh.flatten()), axis=-1).T
    coll_all = np.array([collocation_data[:, i::len(t_col)] for i in range(len(t_col))])
    _, _, _, p_pred = PINN.test(tfnp(collocation_data))
    ppred_all = np.array([p_pred[i::len(t_col)] for i in range(len(t_col))])
    fig, axes = plt.subplots(nrows=len(t_col) // 3, ncols=3,
                             sharex=True, sharey=True,
                             figsize=(6, 8))

    p_pred_minmax = (
        ppred_all.real.min() - 0.15 * ppred_all.real.min(), ppred_all.real.max() - 0.15 * ppred_all.real.max())
    for i, ax in enumerate(axes.flatten(order='F')):
        ax, im = plot_sf(ppred_all[i], coll_all[i, 0], coll_all[i, 1],
                         ax=ax, name='t = {:.3f}s'.format(coll_all[i, 2, 0]),
                         clim=p_pred_minmax, normalise=False)
        if np.logical_and(i == 0, grids is not None):
            grid_m = grids['grid_measured']
            grid_i = grids['grid_initial']
            ax.scatter(grid_m[0], grid_m[1], c='k', s=1, label='measurements')
            ax.scatter(grid_i[0], grid_i[1], c='b', s=1, label='initial conditions')
            ax.legend()
        if (i + 1) % 5 != 0:
            ax.set_xlabel('')
    fig.suptitle('Predicted Pressure')
    fig.subplots_adjust(right=0.8)
    # pressure colorbar
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    cbar_ax.set_title('[Pa]', va='bottom')
    fig.colorbar(im, cax=cbar_ax)
    # fig.tight_layout()
    return fig, None, None


def plot_with_errors(collocation_data, PINN, rirdata):
    Nplots = collocation_data.shape[0]
    Pred_pressure = []
    error_vecs = []
    mean_square_errors = []
    square_errors = []
    tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)
    for n in range(Nplots):
        error_vec, mean_square_error, square_error, p_pred = PINN.test(tfnp(collocation_data[n]), tfnp(rirdata[:, n]))
        Pred_pressure.append(p_pred)
        error_vecs.append(error_vec)
        mean_square_errors.append(mean_square_error)
        square_errors.append(square_error)
    fig, axes = plt.subplots(nrows=3, ncols=Nplots, sharex=True, sharey=True)
    error_vec_minmax = (np.array(error_vecs).min(),
                        np.minimum(np.array(error_vecs).max(),
                                   np.maximum(1., np.array(error_vecs).min() + 1e-5)))
    p_pred_minmax = (np.array(Pred_pressure).min(), np.array(Pred_pressure).max() + np.finfo(np.float32).eps)
    p_true_minmax = (rirdata.min(), rirdata.max() + np.finfo(np.float32).eps)
    for i, ax in enumerate(axes[0]):
        if i == 2:
            name = 'Predicted - \n'
        else:
            name = ''
        ax, im = plot_sf(Pred_pressure[i], collocation_data[i, 0], collocation_data[i, 1],
                         ax=ax, name=name + 't = {:.3f}s'.format(collocation_data[i, 2, 0]),
                         clim=p_pred_minmax, normalise=False)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[1]):
        if i == 2:
            name = 'True'
        else:
            name = ''
        ax, im2 = plot_sf(rirdata[:, i], collocation_data[i, 0], collocation_data[i, 1],
                          ax=ax, name=name,
                          clim=p_true_minmax, normalise=False)
        if i != 0:
            ax.set_ylabel('')
    for i, ax in enumerate(axes[2]):
        if i == 2:
            name = 'Relative Error'
        else:
            name = ''
        ax, im3 = plot_sf(error_vecs[i], collocation_data[i, 0], collocation_data[i, 1],
                          ax=ax, name=name, clim=error_vec_minmax, cmap='hot', normalise=False)
        if i != 0:
            ax.set_ylabel('')
    fig.subplots_adjust(right=0.8)
    # pressure colorbar
    cbar_ax = fig.add_axes([0.82, 0.71, 0.02, 0.2])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax2 = fig.add_axes([0.82, 0.41, 0.02, 0.2])
    fig.colorbar(im2, cax=cbar_ax2)
    # error colorbar
    cbar_ax3 = fig.add_axes([0.82, 0.11, 0.02, 0.2])
    fig.colorbar(im3, cax=cbar_ax3)

    return fig, np.array(mean_square_errors), np.array(square_errors).sum(axis=0).mean(),


def plot_results(collocation_data, PINN, rirdata=None, bounds=None, grids=None):
    if rirdata is None:
        return plot_without_errors(bounds, PINN, grids)
    else:
        return plot_with_errors(collocation_data, PINN, rirdata)


def plot_function_vs_time(data, fs=8000, ax=None, name=None, color=None,
                          linewidth=None, linestyle=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()
    if name is not None:
        ax.set_title(name)
    if linewidth is None:
        linewidth = 1
    if linestyle is None:
        linestyle = '-'
    if color is None:
        color = 'k'
    if label is None:
        label = name
    ax.grid(':', which='both')
    ax.plot(np.arange(len(data)) / fs,
            data,
            color=color,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle)
    return ax


def plot_rir_training(collocation_data, rirdata, PINN, grid=None, plot_frf=True):
    tfnp = lambda x: torch.from_numpy(x).float().to(PINN.device)
    _, _, _, p_pred = PINN.test(tfnp(collocation_data))
    if grid is not None:
        if plot_frf:
            # plot [2, 2] grid with rir, frf and location of grid points
            fig = plt.figure(figsize=(7, 7))
            gs = mpl.gridspec.GridSpec(2, 2, width_ratios=[2, 1])
            ax = plt.subplot(gs[0, 0])
            # make grid_ax take up all the second column
            grid_ax = plt.subplot(gs[:, 1])
            frf_ax = plt.subplot(gs[1, 0])
        else:
            fig = plt.figure(figsize=(7, 3))
            gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[2, 1])
            ax = plt.subplot(gs[0])
            grid_ax = plt.subplot(gs[1])
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    name = 'Predicted - \n'
    ax = plot_function_vs_time(rirdata, ax=ax, name=name, color='k', linewidth=1.5, linestyle='-', label='True')
    ax = plot_function_vs_time(p_pred, ax=ax, name=name, color='seagreen', linewidth=1, linestyle='-',
                               label='Predicted')
    ax.legend(loc="lower right")
    ax.set_xlabel('time [s]')
    ax.set_ylabel('pressure [Pa]')
    if plot_frf:
        # plot frequency response function
        # pad rirdata and p_pred with zeros 2x the length of the rir
        rirdata = np.pad(rirdata, (0, 2 * len(rirdata)), 'constant')
        p_pred = np.pad(p_pred, (0, 2 * len(p_pred)), 'constant')
        both_rirs = np.concatenate((rirdata.reshape(1, -1), p_pred.reshape(1, -1)), axis=0)
        plot_frequency_response(both_rirs, fs=8000, ax=frf_ax)
    if grid is not None:
        # make grid_ax take up all the second column
        grid_ax.set_aspect('equal')
        grid_ax.set_title('Grid points')
        grid_ax.scatter(grid[0], grid[1], marker='.', color='k', label='measurements', s=5)
        grid_ax.scatter(collocation_data[0, 0], collocation_data[1, 0], marker='x',
                        color='r', label='prediction point', s=10)
        grid_ax.set_xlabel('x [m]')
        grid_ax.set_ylabel('y [m]')
    return fig


#  Deep Neural Network
class DNN(nn.Module):
    def __init__(self, n_hidden_layers, lb, ub, siren=True, scale_input=True,
                 activation='sine', n_hidden_features=512, network_type='mlp'):
        super().__init__()  # call __init__ from parent class
        self.siren = siren
        'activation function'
        if self.siren:
            self.activation = nn.Identity()
        else:
            self.activation = nn.Tanh()
        self.lb = lb
        self.ub = ub
        # assert network_type is in ['resnet', 'mlp', 'attention_net']
        self.network_type = network_type
        # assertion
        self.mlp = network_type == 'mlp'
        self.resnet = network_type == 'resnet'
        self.attention_net = network_type == 'attention_net'
        self.n_hidden_layers = n_hidden_layers
        self.attention_net = not self.resnet
        self.n_hidden_layers = n_hidden_layers
        self.scaling = lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        self.scale_input = scale_input
        self.activation_fn = activation
        self.n_hidden_features = n_hidden_features
        in_features = 3
        out_features = 1
        'Initialize neural network as a list using nn.Modulelist'
        if self.siren:
            if self.resnet:
                self.net = SirenResnet(dim_in=in_features,
                                       dim_out=1,
                                       num_resnet_blocks=n_hidden_layers,
                                       num_layers_per_block=3,
                                       num_neurons=self.n_hidden_features)
            elif self.mlp:
                # self.net = SingleBVPNet(
                #     in_features =in_features,  # input dimension, ex. 2d coor
                #     hidden_features=self.n_hidden_features,  # hidden dimension
                #     out_features=out_features,  # output dimension, ex. rgb value
                #     num_hidden_layers=n_hidden_layers,  # number of layers
                # )
                self.net = SirenNet(
                    dim_in=in_features,  # input dimension, ex. 2d coor
                    dim_hidden=self.n_hidden_features,  # hidden dimension
                    dim_out=out_features,  # output dimension, ex. rgb value
                    num_layers=n_hidden_layers + 1,  # number of layers
                    final_activation=nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
                    # different signals may require different omega_0 in the first layer - this is a hyperparameter
                    w0=30.,
                    w0_initial=30.
                )
            elif self.attention_net:
                self.net = MLPWithAttention(input_size=in_features,
                                            hidden_size=self.n_hidden_features,
                                            output_size=out_features,
                                            num_hidden_layers=n_hidden_layers,
                                            siren=True,
                                            positional_encoding=False)
        else:
            if self.resnet:
                self.net = DenseResNet(dim_in=in_features,
                                       dim_out=1,
                                       num_resnet_blocks=n_hidden_layers,
                                       num_layers_per_block=3,
                                       num_neurons=self.n_hidden_features)
            elif self.mlp:
                layers = np.array([3] + n_hidden_layers * [100] + [1])
                self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
                'Xavier Normal Initialization'
                for i in range(len(layers) - 1):
                    nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
                    # set biases to zero
                    nn.init.zeros_(self.linears[i].bias.data)
            elif self.attention_net:
                self.net = MLPWithAttention(input_size=in_features,
                                            hidden_size=self.n_hidden_features,
                                            output_size=out_features,
                                            num_hidden_layers=n_hidden_layers,
                                            siren=True,
                                            positional_encoding=True)

    'foward pass'

    def forward(self, input):
        # batch_size = input.shape[1]
        g = input.clone()
        x, y, t = g[:, 0, :].flatten(), g[:, 1, :].flatten(), g[:, 2, :].flatten()

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        if torch.is_tensor(y) != True:
            y = torch.from_numpy(y)
        if torch.is_tensor(t) != True:
            t = torch.from_numpy(t)

        # convert to float
        x = x.float()
        y = y.float()
        t = t.float()

        # preprocessing input - feature scaling function
        input_preprocessed = torch.stack((x, y, t), dim=-1).view(-1, 3)
        if self.scale_input:
            z = self.scaling(input_preprocessed)
        else:
            z = input_preprocessed
        p_out = self.net(z)  # z: [batchsize x 3]

        return p_out


def fib_sphere(num_points, radius=1):
    ga = (3 - np.sqrt(5.)) * np.pi  # golden angle

    # Create a list of golden angle increments along tha range of number of points
    theta = ga * np.arange(num_points)

    # Z is a split into a range of -1 to 1 in order to create a unit circle
    z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

    # a list of the radii at each height step of the unit circle
    alpha = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles
    y = alpha * np.sin(theta)
    x = alpha * np.cos(theta)

    x_batch = np.tensordot(radius, x, 0)
    y_batch = np.tensordot(radius, y, 0)
    z_batch = np.tensordot(radius, z, 0)

    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch, y_batch, z_batch, s = 20)
    # # ax.scatter(x, y, z , s = 3)
    # plt.show()
    return np.vstack([x_batch, y_batch, z_batch])


# get square grid
def get_square_grid(xmin=-1, xmax=1, ymin=-1, ymax=1, num_points=35):
    x = np.linspace(xmin, xmax, num_points)
    y = np.linspace(ymin, ymax, num_points)

    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    z_grid = np.zeros_like(x_grid)

    return np.vstack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])


def plot_single_frf(rir, fs=8000, ax=None, color='k'):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    rirdata = np.pad(rir, (0, 2 * len(rir)), 'constant')
    plot_frequency_response(rirdata, fs=fs, ax=ax, color=color)
    return ax


def plane_wave_sensing_matrix(sph_grid, mic_pos, t, c=343.0):
    """
    Compute the plane wave sensing matrix using a sinc function approximation of the Dirac delta function.

    Parameters:
        sph_grid: np.ndarray
            A (3, N) array representing the spherical grid of plane wave origins.
        mic_pos: np.ndarray
            A (3, M) array representing the microphone positions.
        c: float
            The speed of sound (default: 343 m/s).
        T: float
            The duration of the impulse response (default: 1 ms).
        fs: float
            The sampling frequency (default: 44100 Hz).

    Returns:
        np.ndarray
            A (M, N, T_samples) sensing matrix consisting of plane waves in the time domain.
    """

    # Compute the distance between each microphone and plane wave origin
    r = np.sqrt(np.sum((mic_pos[:, np.newaxis, :] - sph_grid[:, :, np.newaxis]) ** 2, axis=0))
    T = 1e-3
    fs = 8e3
    T_samples = int(T * fs)
    t = np.linspace(-T / 2, T / 2, T_samples, endpoint=False)
    sincarg = r[np.newaxis, :, :] / c - t[np.newaxis, np.newaxis, :]

    # Compute the time-domain plane wave using a sinc function approximation of the Dirac delta function
    H = np.sinc(t[-1] - r / c)
    return H


def plane_wave_sensing_matrix(sph_grid, mic_pos, t_obs, c=343.0):
    """
    Compute the plane wave sensing matrix using a sinc function approximation of the Dirac delta function.
    """
    r = np.sqrt(np.sum((mic_pos[:, np.newaxis, :] - sph_grid[:, :, np.newaxis]) ** 2, axis=0))
    delays = r / c
    fs = 8000
    t = np.arange(0, t_obs, 1 / fs)
    starting_time = 0
    base_time = t - starting_time
    # Compute the time-domain plane wave using a sinc function approximation of the Dirac delta function
    H = np.sinc(fs * (base_time[:, np.newaxis] - delays))
    plot_sf(100 * H[2], mic_pos[0], mic_pos[1], colorbar=True, normalise=False, interpolated=False)
    plt.show()


def single_plane_wave_time_domain(direction, mic_pos, t, c=343.0):
    """
    Singe plane wave in the time domain using a sinc function approximation of the Dirac delta function
    """
    # direction is a vector with azimuth and elevation
    # mic_pos is a matrix with the microphone positions
    # t is the time vector
    # c is the speed of sound

    # direction to cartesian coordinates
    direction = np.deg2rad(direction)
    y = np.cos(direction[1]) * np.cos(direction[0])
    x = np.sin(direction[1]) * np.cos(direction[0])
    z = np.sin(direction[0])

    # compute the distance between the source and the microphones
    r = np.sqrt(np.sum((mic_pos - np.array([x, y, z])[:, np.newaxis]) ** 2, axis=0))
    delays = r / c
    fs = 8000
    t = np.arange(0, 1, 1 / fs)
    starting_time = 0
    base_time = t - starting_time
    # Compute the time-domain plane wave using a sinc function approximation of the Dirac delta function
    H = np.sinc(fs * (base_time[:, np.newaxis] - delays))
    # truncate H so it only includes the time of the impulse response within the microphone array
    t_init = np.argmin(np.abs(t - delays[-1]))
    t_end = np.argmin(np.abs(t - delays[0]))
    H = H[t_init:t_end, :]
    plot_sf(H[0], mic_pos[0], mic_pos[1], colorbar=True, normalise=False, interpolated=False)
    plt.show()


def point_source_response(t, c, fs, source_coords, receiver_coords, n_iter):
    assert n_iter >= 1
    # if source_coords.shape[-1] == 3:
    #     source_coords = source_coords.T
    # if receiver_coords.shape[-1] == 3:
    #     receiver_coords = receiver_coords.T

    if source_coords.size == 0:
        return np.zeros((receiver_coords.shape[0]))
    else:
        d = distance_between(source_coords, receiver_coords)

        Dt = t[n_iter - 1] - (d / c)

        # attenuation constant
        m = 0.039144
        air_attenuation = np.exp(-m * d)
        # source strengths
        if n_iter == 1:
            An = (2 * np.random.randint(0, 2, size=(Dt.shape[0])) - 1)
        else:
            # An = np.random.uniform(-1/(n_iter), 1/(n_iter), (Dt.shape[0], 1))
            An = np.random.uniform(-1, 1, (Dt.shape[0], 1))

        responses = np.sum(An * air_attenuation / (4 * np.pi * d) * np.sinc(fs * Dt), axis=0)
        return responses


def distance_between(s, r):
    """Distance of all combinations of locations in s and r
    Args:
        s (ndarray [N, 2]): cartesian coordinates of s
        r (ndarray [M, 2]): cartesian coordinates of r
    Returns:
        ndarray [M, N]: distances in meters
    """
    s = torch.atleast_2d(s)
    r = torch.atleast_2d(r)
    if s.shape[-1] != 2:
        s = s.T
    if r.shape[-1] != 2:
        r = r.T
    return torch.linalg.norm(s[None, :] - r[:, None], axis=-1)


def plane_wave(t, c, fs, source_coords, receiver_coords):
    """

    Parameters
    ----------
    t - instanteneous time (scalar)
    c
    fs
    source_coords
    receiver_coords

    Returns
    -------

    """
    with torch.no_grad():
        receiver_coords = receiver_coords / torch.linalg.norm(receiver_coords, axis=0)
    d = distance_between(source_coords, receiver_coords)
    Dt = t[:, None] - (d / c)
    return torch.sinc(fs * Dt)


#  PINN
# https://github.com/alexpapados/Physics-Informed-Deep-Learning-Solid-and-Fluid-Mechanics

def PINN_LBFGS(model, batch, device, optimizer):
    """
    Train a model using the LBFGS optimizer
    """
    # dnn = model.dnn.to(device)
    data_input = batch['collocation_train']
    pde_input = batch['collocation_pde']
    ic_input = batch['collocation_ic']
    p_data = batch['pressure_batch']
    p_ic = torch.zeros_like(p_data)
    data_loss_weights = batch['data_loss_weights']

    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        (loss_val, loss_p,
         loss_f, loss_bc,
         loss_ic, norm_ratio,
         std_ratio, maxabs_ratio) = model.loss(data_input.to(device),
                                               pde_input.to(device),
                                               ic_input.to(device),
                                               p_data.to(device),
                                               p_ic.to(device),
                                               data_loss_weights.to(device)
                                               )
        if loss_val.requires_grad:
            loss_val.backward()
        print("LBFGS-B: PINN Loss {}".format(loss_val.item()))
        return loss_val

    optimizer.step(closure)


class FCN():
    def __init__(self,
                 bounds,
                 n_hidden_layers=4,
                 device=None,
                 siren=True,
                 lambda_data=1.,
                 lambda_pde=1e-4,
                 lambda_bc=1e-2,
                 lambda_ic=1e-2,
                 loss_fn='mae',
                 c=343.,
                 output_scaler=None,
                 fs=48e3,
                 map_input=True,
                 activation='sine',
                 n_hidden_features=512,
                 sigmas=[None],
                 network_type='mlp'):
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        if loss_fn in ['mse', 'MSE']:
            self.loss_function = nn.MSELoss(reduction='mean')
            # self.loss_function = lambda y_hat, y : ((torch.abs(y - y_hat)**2)/torch.abs(y)**2).mean()
            self.loss_function_pde = nn.MSELoss(reduction='mean')
        else:
            self.loss_function = nn.L1Loss(reduction='mean')
            self.loss_function_pde = nn.L1Loss(reduction='mean')
            # self.loss_function = nn.L1Loss(reduction='sum')
        'Initialize iterator'
        self.iter = 0
        self.fs = fs
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.siren = siren
        self.activation = activation
        self.n_hidden_features = n_hidden_features
        self.sigmas = sigmas
        self.network_type = network_type
        'speed of sound'
        self.c = c
        self.output_scaler = output_scaler if output_scaler is not None else nn.Identity()

        (self.xmin, self.xmax) = bounds['x']
        (self.ymin, self.ymax) = bounds['y']
        (self.tmin, self.tmax) = bounds['t']
        (self.xdata_min, self.xdata_max) = bounds['xdata']
        (self.ydata_min, self.ydata_max) = bounds['ydata']

        self.tmax *= self.c
        self.tmin *= self.c
        self.lb = torch.Tensor([self.xmin, self.ymin, self.tmin]).to(self.device)
        self.ub = torch.Tensor([self.xmax, self.ymax, self.tmax]).to(self.device)
        'Call our DNN'
        self.dnn = DNN(n_hidden_layers, lb=self.lb, ub=self.ub, siren=siren,
                       scale_input=map_input, network_type=self.network_type,
                       activation=self.activation, n_hidden_features=self.n_hidden_features).to(device)
        'test with cosine similarity loss'
        self.cosine_sim = nn.CosineEmbeddingLoss()
        # self.ansatz_formulation = ansatz_formulation
        # if self.ansatz_formulation:
        #     # self.r_source = torch.nn.Parameter(torch.randn(1000, 2, requires_grad=True).float().to(self.device))
        #     self.r_source = torch.rand(1024, 2, requires_grad=True, device=self.device, dtype= torch.float32)

    def cylindrical_coords(self, input):
        x, y, t = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y, x)
        return r, phi

    def loss_data(self, input, pm, data_loss_weights=None):
        g = input.clone()
        g = self.scale_t(g)
        g.requires_grad = True
        # if self.ansatz_formulation:
        #     pressure = (self.dnn(g)*plane_wave(g[0,2],
        #                                        1.,
        #                                        self.fs/self.c,
        #                                        self.r_source,
        #                                        g[0, :2])).sum(axis = -1).unsqueeze(0)
        # else:
        pressure = self.dnn(g).T

        loss_u = self.loss_function(data_loss_weights * pressure, data_loss_weights * pm)
        # loss_u = self.loss_function(pressure, pm)
        norm_ratio = pm.norm() / pressure.norm()
        std_ratio = pm.std() / pressure.std()
        maxabs_ratio = abs(pm).max() / abs(pressure).max()

        return loss_u, norm_ratio, std_ratio, maxabs_ratio

    def loss_PDE(self, input):

        g = input.clone()
        g = self.scale_t(g)
        g.requires_grad = True

        # if self.ansatz_formulation:
        #     pressure = (self.dnn(g)*plane_wave(g[0,2],
        #                                        1.,
        #                                        self.fs/self.c,
        #                                        self.r_source,
        #                                        g[0, :2])).sum(axis = -1).unsqueeze(0)
        # else:
        pnet = self.dnn(g).T

        # pnet = self.output_scaler.backward_rir(pressure)

        p_r_t = autograd.grad(pnet.view(-1, 1), g, torch.ones([input.view(-1, 3).shape[0], 1]).to(self.device),
                              create_graph=True)[0]
        p_rr_tt = \
            autograd.grad(p_r_t.view(-1, 1), g, torch.ones(input.view(-1, 1).shape).to(self.device),
                          create_graph=True)[0]
        p_xx = p_rr_tt[:, [0]]
        p_yy = p_rr_tt[:, [1]]
        p_tt = p_rr_tt[:, [2]]

        # given that x, y are scaled here so that x' = x/c and y' = y/c, then c = 1
        # f = p_tt - self.c * (p_xx + p_yy)
        f = p_xx + p_yy - 1. * p_tt

        loss_f = self.loss_function_pde(f.view(-1, 1), torch.zeros_like(f.view(-1, 1)))

        return loss_f

    def loss_bc(self, input):
        g = input.clone()
        g = self.scale_t(g)
        g.requires_grad = True

        pnet = self.dnn(g).T
        p_x_y_t = autograd.grad(pnet.view(-1, 1), g, torch.ones([input.view(-1, 3).shape[0], 1]).to(self.device),
                                create_graph=True)[0]

        # Apply the radiation conditions (open boundaries)
        # dp/dt - c dp/dx = 0  x = xmin , dp/dt + c dp/dx = 0  x = xmax
        # dp/dt + c dp/dy = 0  y = ymax , dp/dt - c dp/dy = 0  y = ymin
        top_indx = (g[0, 1] == self.ydata_max).nonzero().flatten()
        bottom_indx = (g[0, 1] == self.ydata_min).nonzero().flatten()
        left_indx = (g[0, 0] == self.xdata_min).nonzero().flatten()
        right_indx = (g[0, 0] == self.xdata_max).nonzero().flatten()
        dp_dt = p_x_y_t[:, [2]].view(-1)
        dp_dx = p_x_y_t[:, [0]].view(-1)
        dp_dy = p_x_y_t[:, [1]].view(-1)
        f = torch.zeros_like(dp_dt)
        f[top_indx] = dp_dt[top_indx] + self.c * dp_dy[top_indx]
        f[bottom_indx] = dp_dt[bottom_indx] - self.c * dp_dy[bottom_indx]
        f[left_indx] = dp_dt[left_indx] - self.c * dp_dx[left_indx]
        f[right_indx] = dp_dt[right_indx] + self.c * dp_dx[right_indx]
        bcs_loss = self.loss_function_pde(f.view(-1, 1), torch.zeros_like(f.view(-1, 1)))
        return bcs_loss

    def loss_ic(self, input, p_ic=None):
        g = input.clone()
        g = self.scale_t(g)
        g.requires_grad = True
        pnet = self.dnn(g).T
        if p_ic is None:
            p_x_y_t = autograd.grad(pnet.view(-1, 1), g, torch.ones([input.view(-1, 3).shape[0], 1]).to(self.device),
                                    create_graph=True)[0]
            dp_dt = p_x_y_t[:, [2]].flatten()
            f = pnet + dp_dt
            ics_loss = self.loss_function_pde(f.view(-1, 1), torch.zeros_like(f.view(-1, 1)))
        else:
            ics_loss = self.loss_function_pde(pnet.view(-1, 1), p_ic.view(-1, 1))
        return ics_loss

    def loss(self, input_data, input_pde, input_bc, input_ic, pm, p_ic, data_loss_weights, iter=None):
        if self.output_scaler is not None:
            pm = self.output_scaler.forward_rir(pm)
        if torch.is_nonzero(p_ic.sum()):
            if self.output_scaler is not None:
                p_ic = self.output_scaler.forward_rir(p_ic)
            loss_ic = self.loss_ic(input_ic, p_ic)
        else:
            loss_ic = torch.tensor(0.).to(self.device)

        loss_p, norm_ratio, std_ratio, maxabs_ratio = self.loss_data(input_data, pm, data_loss_weights)
        loss_f = self.loss_PDE(input_pde)
        loss_bc = self.loss_bc(input_bc)
        if None in self.sigmas:
            # weight the loss based on the number of samples
            w_data = input_data.shape[0] / (input_data.shape[0] + input_pde.shape[0] + input_ic.shape[0])
            w_pde = input_pde.shape[0] / (input_data.shape[0] + input_pde.shape[0] + input_ic.shape[0])
            w_ic = input_ic.shape[0] / (input_data.shape[0] + input_pde.shape[0] + input_ic.shape[0])
            w_bc = input_bc.shape[0] / (input_data.shape[0] + input_pde.shape[0] + input_ic.shape[0])
            total_loss = self.lambda_data * w_data * loss_p + w_pde * loss_f + self.lambda_bc * w_bc * loss_bc \
                         + self.lambda_ic * w_ic * loss_ic
        else:
            loss_bucket = [loss_p, loss_f, loss_bc, loss_ic]
            total_loss = 0
            for o, sig in enumerate(self.sigmas):
                total_loss += loss_bucket[o] / (2 * sig.pow(2)) + torch.log(sig)
            total_loss *= 1e3
        return total_loss, loss_p, loss_f, loss_bc, loss_ic, norm_ratio, std_ratio, maxabs_ratio

    'callable for optimizer'

    def closure(self, optimizer, train_input, test_input, p_train, p_test):

        optimizer.zero_grad()

        loss = self.loss(train_input, p_train)

        loss.backward()

        self.iter += 1

        if self.iter % 100 == 0:
            error_vec, _, _ = self.test(test_input, p_test)
            # TODO: FIX HERE
            print(
                'Relative Error (Test): %.5f' %
                (
                    error_vec.cpu().detach().numpy(),
                )
            )

        return loss

    def SGD_step(self, data_input, pde_input, bc_input, ic_input, p_data, p_ic, data_loss_weights=None, iter=None):

        loss, loss_data, loss_pde, loss_bc, loss_ic, norm_ratio, std_ratio, maxabs_ratio \
            = self.loss(data_input, pde_input, bc_input, ic_input, p_data, p_ic, data_loss_weights, iter)

        loss.backward()

        return (loss.cpu().detach().numpy(),
                loss_data.cpu().detach().numpy(),
                loss_pde.cpu().detach().numpy(),
                loss_bc.cpu().detach().numpy(),
                loss_ic.cpu().detach().numpy(),
                norm_ratio.cpu().detach().numpy(),
                std_ratio.cpu().detach().numpy(),
                maxabs_ratio.cpu().detach().numpy()
                )

    'test neural network'

    def test(self, test_input, p_true=None):
        g = test_input.clone().unsqueeze(0)
        g = self.scale_t(g)
        p_pred = self.dnn(g).T
        if self.output_scaler is not None:
            p_pred = self.output_scaler.backward_rir(p_pred)
        if p_true is not None:
            sq_err = torch.abs(p_true - p_pred.squeeze(0)) ** 2
            mse = sq_err.mean().item()
            sq_err = sq_err.cpu().detach().numpy()
            # Error vector
            error_vec = torch.abs((p_true - p_pred.squeeze(0)) ** 2 / (p_true + np.finfo(np.float32).eps) ** 2)
            error_vec = error_vec.cpu().detach().numpy()
        else:
            error_vec = None
            mse = None
            sq_err = None
        p_pred = p_pred.squeeze(0).cpu().detach().numpy()

        return error_vec, mse, sq_err, p_pred

    def inference(self, input):
        g = input.clone()
        g = self.scale_t(g)
        # if self.ansatz_formulation:
        #     p_pred = (self.dnn(g)*plane_wave(g[0,2],
        #                                        1.,
        #                                        self.fs/self.c,
        #                                        self.r_source,
        #                                        g[0, :2])).sum(axis = -1).unsqueeze(0)
        # else:
        p_pred = self.dnn(g).T
        # if self.output_scaler is not None:
        #     p_pred = self.output_scaler.backward_rir(p_pred)
        return p_pred

    def scale_xy(self, input):
        x, y, t = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        x = x / self.c
        y = y / self.c
        return torch.stack((x, y, t), dim=-1).view(3, -1).unsqueeze(0)

    def scale_t(self, input):
        x, y, t = input[:, 0, :].flatten(), input[:, 1, :].flatten(), input[:, 2, :].flatten()
        t = t * self.c
        return torch.vstack((x, y, t)).unsqueeze(0)


def get_T60(edc, fs=8000):
    """ Function to get the reverberation time from an EDC.
    Args:
        edc (np.ndarray): EDC with shape (n_samples,).
    Returns:
        float: Reverberation time in seconds.
    """
    from scipy import stats
    # convert to dB
    edc_dB = 10 * np.log10(edc / np.max(edc) + np.finfo(np.float32).eps)
    # Linear regression parameters for computing RT60
    init_db = -5
    end_db = -25
    factor = 3.0
    t = np.arange(edc_dB.shape[0]) / fs

    energy_init = edc_dB[np.abs(edc_dB - init_db).argmin()]
    energy_end = edc_dB[np.abs(edc_dB - end_db).argmin()]
    init_sample = np.where(edc_dB == energy_init)[0][0]
    end_sample = np.where(edc_dB == energy_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = edc_dB[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]
    # line = slope * t + intercept
    db_regress_init = (init_db - intercept) / slope
    db_regress_end = (end_db - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)
    return t60


def get_avg_edc(rirs):
    """ Function to get the average (normalised) EDC from an array of RIRs.
    Args:
        rirs (np.ndarray): Array of RIRs with shape (n_rirs, n_samples).
    Returns:
        np.ndarray: Average (normalised) EDC with shape (n_samples,).
    """
    # backward schroeder integration
    edc = np.flip(np.cumsum(np.flip(rirs ** 2, axis=1), axis=1), axis=1)
    # calculate average reverberation time from each edc with linear regression
    T60 = np.zeros(edc.shape[0])
    for i in range(edc.shape[0]):
        T60[i] = get_T60(edc[i, :])

    edc_mean = np.mean(edc, axis=0)
    edc_mean /= np.max(edc_mean)
    return edc_mean


def sample_boundary_condition_coords(grid_bounds, n_points):
    """ Function to sample coordinates over grid boundary, given a square or rectangle grid."""
    # sample x and y coordinates
    x_min, x_max, y_min, y_max = grid_bounds
    x_bc = lhs(1, samples=n_points // 2)
    y_bc = lhs(1, samples=n_points // 2)
    # scale the coordinates to the grid bounds
    x_bc = x_bc * (x_max - x_min) + x_min
    # get n_points//4 at y_min and n_points//4 at y_max
    y_bc_x = np.concatenate((np.ones(n_points // 4) * y_min, np.ones(n_points // 4) * y_max))
    y_bc = y_bc * (y_max - y_min) + y_min
    # get n_points//4 at x_min and n_points//4 at x_max
    x_bc_y = np.concatenate((np.ones(n_points // 4) * x_min, np.ones(n_points // 4) * x_max))
    # concatenate the coordinates
    xy_top_bottom = np.concatenate((x_bc.T, y_bc_x.reshape(1, -1)), axis=0)
    xy_left_right = np.concatenate((x_bc_y.reshape(1, -1), y_bc.T), axis=0)
    # concatenate the coordinates and shuffle
    xy_bc = np.concatenate((xy_top_bottom, xy_left_right), axis=1)
    # # scatter plot the coordinates
    # plt.scatter(xy_top_bottom[0, :], xy_top_bottom[1, :], color = 'r', label = 'top bottom')
    # plt.scatter(xy_left_right[0, :], xy_left_right[1, :], color = 'g', label = 'left right')
    # plt.legend()
    # plt.show()
    return xy_bc


class PINNDataset_w_IC(Dataset):
    def __init__(self,
                 measured_data,
                 ic_data,
                 x_ic,
                 y_ic,
                 x_m,
                 y_m,
                 t,
                 t_ind,
                 n_pde_samples=800,
                 counter=1,
                 maxcounter=1e5,
                 curriculum_training=False,
                 t_weighting_factor=False,
                 batch_size=300,
                 bounds=None):
        self.tfnp = lambda x: torch.from_numpy(x).float()
        self.curriculum_training = curriculum_training
        self.counter = counter
        self.maxcounter = maxcounter
        # self.maxcounter = -1
        self.TrainData = measured_data
        self.n_pde_samples = n_pde_samples
        # self.BCData = refdata[x_y_boundary_ind]
        self.t_ind = t_ind
        self.batch_size = batch_size
        # self.x_y_boundary_ind = x_y_boundary_ind
        self.x_m = x_m
        self.y_m = y_m
        self.x_ic = x_ic
        self.y_ic = y_ic
        self.t = t
        self.tt = np.repeat(self.t, len(self.x_m))
        self.xx = np.tile(self.x_m, len(self.t))
        self.yy = np.tile(self.y_m, len(self.t))
        self.collocation_all = self.tfnp(np.stack([self.xx, self.yy, self.tt], axis=0))
        self.pressure_all = self.tfnp(measured_data[:, self.t_ind].flatten())
        self.ic_data = ic_data[:, :120]
        self.t_ic = self.t[:120]
        self.counter_fun = lambda x, n: int(n * x)
        decay_rate = np.linspace(0, 1, len(self.t))
        if t_weighting_factor is None:
            #     self.t_weight = 10 * (1 - .98) ** decay_rate
            # else:
            self.t_weight = np.ones_like(decay_rate)
        else:
            self.t_weight = t_weighting_factor
        if bounds is None:
            self.xmax = self.x_ref.max()
            self.xmin = self.x_ref.min()
            self.ymax = self.y_ref.max()
            self.ymin = self.y_ref.min()
            self.tmax = self.t[self.t_ind].max()
        else:
            self.xmax = bounds['x'][0]
            self.xmin = bounds['x'][1]
            self.ymax = bounds['y'][0]
            self.ymin = bounds['y'][1]
            self.tmax = bounds['t'][1]

    def __len__(self):
        return 1
        # return len(self.t_ind)

    def __getitem__(self, idx):
        if np.logical_and(self.curriculum_training, self.counter < self.maxcounter):
            sample_limit = self.counter_fun(self.counter / self.maxcounter, len(self.t_ind))
            sample_limit = np.maximum(self.batch_size, sample_limit)
            idx = np.random.randint(0, sample_limit, self.batch_size)
            t_batch_indx = self.t_ind[idx]
            t_ind_temp = self.t_ind[:sample_limit]
            t_lims = (self.t[t_ind_temp].min(), self.t[t_ind_temp].max())
        elif np.logical_and(not self.curriculum_training, self.counter < self.maxcounter):
            window_size = self.batch_size
            overlap = self.batch_size // 2
            t_ind_windowed = window(self.t_ind, w=window_size, o=overlap)  # 100 taps, 25 overlap
            n_windows = t_ind_windowed.shape[0]
            window_number = self.counter_fun(self.counter / self.maxcounter, n_windows)
            # t_ind_temp = self.t_ind[:(progressive_t_counter + 1)]
            t_ind_temp = t_ind_windowed[window_number]
            # idx = np.random.randint(0, progressive_t_counter + 1)
            idx = np.random.randint(0, window_size, window_size)
            t_batch_indx = t_ind_temp[idx]
            t_lims = (self.t[t_ind_temp].min(), self.t[t_ind_temp].max())
        else:
            idx = np.random.randint(0, len(self.t_ind), self.batch_size)
            t_batch_indx = self.t_ind[idx]
            t_lims = (self.t[self.t_ind].min(), self.t[self.t_ind].max())
        t_data = self.t[t_batch_indx]
        pressure_batch = self.TrainData[:, t_batch_indx].flatten(order='F')
        pressure_bc_batch = self.TrainData[:, t_batch_indx].flatten(order='F')
        pressure_ic_batch = self.ic_data.flatten(order='F')
        x_data, y_data = self.x_m, self.y_m

        grid_pde = (2 * (lhs(2, self.n_pde_samples)) / 1 - 1)
        x_pde = self.xmax * grid_pde[:, 0]
        y_pde = self.ymax * grid_pde[:, 1]
        x_bc = self.xmax * grid_pde[:, 0]
        y_bc = self.ymax * grid_pde[:, 1]

        if self.counter < self.maxcounter:
            t_pde = t_data.max() * lhs(1, self.n_pde_samples).squeeze(-1)
            t_bc = t_data.max() * lhs(1, self.n_pde_samples).squeeze(-1)
        else:
            t_pde = self.tmax * lhs(1, self.n_pde_samples).squeeze(-1)
            t_bc = t_data.max() * lhs(1, self.n_pde_samples).squeeze(-1)

        data_loss_weights = self.t_weight[t_batch_indx]
        data_loss_weights = np.repeat(data_loss_weights, len(x_data))
        tt_data = np.repeat(t_data, len(x_data))
        xx_data = np.tile(x_data, len(t_data))
        yy_data = np.tile(y_data, len(t_data))
        x_ic = np.tile(self.x_ic, len(self.t_ic))
        y_ic = np.tile(self.y_ic, len(self.t_ic))
        t_ic = np.repeat(self.t_ic, len(self.x_ic))
        collocation_train = np.stack([xx_data, yy_data, tt_data], axis=0)
        collocation_pde = np.stack([x_pde, y_pde, t_pde], axis=0)
        collocation_bc = np.stack([x_bc, y_bc, t_bc], axis=0)
        collocation_ic = np.stack([x_ic, y_ic, t_ic], axis=0)
        self.counter += 1

        return {
            'collocation_train': self.tfnp(collocation_train),
            'collocation_bc': self.tfnp(collocation_bc),
            'collocation_pde': self.tfnp(collocation_pde),
            'collocation_ic': self.tfnp(collocation_ic),
            'pressure_bc_batch': self.tfnp(pressure_bc_batch),
            'pressure_batch': self.tfnp(pressure_batch),
            't_batch_indx': t_batch_indx,
            'max_t': t_data.max(),
            'data_loss_weights': self.tfnp(data_loss_weights),
            't_lims': t_lims,
            'pressure_ic_batch': self.tfnp(pressure_ic_batch)}


class PINNDataset(Dataset):
    def __init__(self,
                 refdata,
                 measured_data,
                 x_ref,
                 y_ref,
                 x_m,
                 y_m,
                 t,
                 t_ind,
                 n_pde_samples=800,
                 counter=1,
                 maxcounter=1e5,
                 curriculum_training=False,
                 t_weighting_factor=False,
                 batch_size=300):
        self.tfnp = lambda x: torch.from_numpy(x).float()
        self.curriculum_training = curriculum_training
        self.counter = counter
        self.maxcounter = maxcounter
        # self.maxcounter = -1
        self.TrainData = measured_data
        self.n_pde_samples = n_pde_samples
        # self.BCData = refdata[x_y_boundary_ind]
        self.t_ind = t_ind
        self.batch_size = batch_size
        # self.x_y_boundary_ind = x_y_boundary_ind
        self.x_m = x_m
        self.y_m = y_m
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.t = t
        self.tt = np.repeat(self.t, len(self.x_ref))
        self.xx = np.tile(self.x_ref, len(self.t))
        self.yy = np.tile(self.y_ref, len(self.t))
        self.collocation_all = self.tfnp(np.stack([self.xx, self.yy, self.tt], axis=0))
        self.pressure_all = self.tfnp(refdata[:, self.t_ind].flatten())
        self.xmax = self.x_ref.max()
        self.xmin = self.x_ref.min()
        self.ymax = self.y_ref.max()
        self.ymin = self.y_ref.min()
        self.tmax = self.t[self.t_ind].max()
        self.counter_fun = lambda x, n: int(n * x)
        decay_rate = np.linspace(0, 1, len(self.t))
        if t_weighting_factor is None:
            #     self.t_weight = 10 * (1 - .98) ** decay_rate
            # else:
            self.t_weight = np.ones_like(decay_rate)
        else:
            self.t_weight = t_weighting_factor
        # self.batch_size = batch_size
        # self.n_time_instances = int(0.6 * self.batch_size)
        # self.n_spatial_instances = self.batch_size - self.n_time_instances
        # self.n_spatial_instances = len(data_ind)
        # self.n_time_instances = self.batch_size - self.n_spatial_instances

    def __len__(self):
        return 1
        # return len(self.t_ind)

    def __getitem__(self, idx):
        if np.logical_and(self.curriculum_training, self.counter < self.maxcounter):
            sample_limit = self.counter_fun(self.counter / self.maxcounter, len(self.t_ind))
            sample_limit = np.maximum(self.batch_size, sample_limit)
            idx = np.random.randint(0, sample_limit, self.batch_size)
            t_batch_indx = self.t_ind[idx]
            t_ind_temp = self.t_ind[:sample_limit]
            t_lims = (self.t[t_ind_temp].min(), self.t[t_ind_temp].max())
        elif np.logical_and(not self.curriculum_training, self.counter < self.maxcounter):
            window_size = self.batch_size
            overlap = self.batch_size // 2
            t_ind_windowed = window(self.t_ind, w=window_size, o=overlap)  # 100 taps, 25 overlap
            n_windows = t_ind_windowed.shape[0]
            window_number = self.counter_fun(self.counter / self.maxcounter, n_windows)
            # t_ind_temp = self.t_ind[:(progressive_t_counter + 1)]
            t_ind_temp = t_ind_windowed[window_number]
            # idx = np.random.randint(0, progressive_t_counter + 1)
            idx = np.random.randint(0, window_size, window_size)
            t_batch_indx = t_ind_temp[idx]
            t_lims = (self.t[t_ind_temp].min(), self.t[t_ind_temp].max())
        else:
            idx = np.random.randint(0, len(self.t_ind), self.batch_size)
            t_batch_indx = self.t_ind[idx]
            t_lims = (self.t[self.t_ind].min(), self.t[self.t_ind].max())
        t_data = self.t[t_batch_indx]
        pressure_batch = self.TrainData[:, t_batch_indx].flatten(order='F')
        pressure_bc_batch = self.TrainData[:, t_batch_indx].flatten(order='F')
        x_data, y_data = self.x_m, self.y_m

        grid_pde = (2 * (lhs(2, self.n_pde_samples)) / 1 - 1)
        grid_ic = (2 * (lhs(2, self.n_pde_samples)) / 1 - 1)
        x_pde = self.xmax * grid_pde[:, 0]
        y_pde = self.ymax * grid_pde[:, 1]
        x_bc, y_bc = sample_boundary_condition_coords([self.xmin, self.xmax, self.ymin, self.ymax], self.n_pde_samples)
        x_ic = self.xmax * grid_ic[:, 0]
        y_ic = self.ymax * grid_ic[:, 1]
        t_ic = np.zeros(self.n_pde_samples)
        if self.counter < self.maxcounter:
            t_pde = t_data.max() * lhs(1, self.n_pde_samples).squeeze(-1)
            t_bc = t_data.max() * lhs(1, self.n_pde_samples).squeeze(-1)
        else:
            t_pde = self.tmax * lhs(1, self.n_pde_samples).squeeze(-1)
            t_bc = t_data.max() * lhs(1, self.n_pde_samples).squeeze(-1)

        data_loss_weights = self.t_weight[t_batch_indx]
        data_loss_weights = np.repeat(data_loss_weights, len(x_data))
        tt_data = np.repeat(t_data, len(x_data))
        xx_data = np.tile(x_data, len(t_data))
        yy_data = np.tile(y_data, len(t_data))
        collocation_train = np.stack([xx_data, yy_data, tt_data], axis=0)
        collocation_pde = np.stack([x_pde, y_pde, t_pde], axis=0)
        collocation_bc = np.stack([x_bc, y_bc, t_bc], axis=0)
        collocation_ic = np.stack([x_ic, y_ic, t_ic], axis=0)
        self.counter += 1

        return {
            'collocation_train': self.tfnp(collocation_train),
            'collocation_bc': self.tfnp(collocation_bc),
            'collocation_pde': self.tfnp(collocation_pde),
            'collocation_ic': self.tfnp(collocation_ic),
            'pressure_bc_batch': self.tfnp(pressure_bc_batch),
            'pressure_batch': self.tfnp(pressure_batch),
            't_batch_indx': t_batch_indx,
            'max_t': t_data.max(),
            'data_loss_weights': self.tfnp(data_loss_weights),
            't_lims': t_lims, }


def window(a, w=4, o=2, copy=False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view


class DenseResNet(nn.Module):
    """
    This is a ResNet Class.
    -> dim_in: network's input dimension
    -> dim_out: network's output dimension
    -> num_resnet_blocks: number of ResNet blocks
    -> num_layers_per_block: number of layers per ResNet block
    -> num_neurons: number of neurons in each layer
    -> activation: Non-linear activations function that you want to use. E.g. nn.Sigmoid(), nn.ReLU()
    -> fourier_features: whether to pass the inputs through Fourier mapping. E.g. True or False
    -> m_freq: how many frequencies do you want the inputs to be mapped to
    -> sigma: controls the spectrum of frequencies.
              If sigma is greater more frequencies are consider.
              You can also look at it as sampling from the standard normal, Z~N(0, 1),
              and mapping to another normal, X~N(\mu, \sigma^2), using x = mu + sigma*z.
    -> tune_beta: do you want to consider the parameter beta in the activation functions in each layer? E.g., Tanh(beta*x).
                  In practice it is observed that training beta (i.e. tune_beta=True) could improve convergence.
                  If tune_beta=False, you get the a fixed beta i.e. beta=1.
    -> The method model_capacity() returns the number of layers and parameters in the network.
    """

    def __init__(self, dim_in=3, dim_out=1, num_resnet_blocks=3,
                 num_layers_per_block=2, num_neurons=128, activation=nn.Tanh(),
                 fourier_features=False, n_freqs=512, sigma_xy=128, sigma_t=4096, tune_beta=False,
                 device='cuda'):
        super(DenseResNet, self).__init__()

        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta
        self.n_freqs = n_freqs
        self.device = device

        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1)).to(self.device)
            self.beta = nn.Parameter(torch.ones(self.num_resnet_blocks, self.num_layers_per_block)).to(self.device)

        else:
            self.beta0 = torch.ones(1, 1).to(self.device)
            self.beta = torch.ones(self.num_resnet_blocks, self.num_layers_per_block).to(self.device)

        if self.fourier_features:
            self.first = nn.Linear(6 * self.n_freqs, num_neurons)
            # self.B = nn.Parameter(sigma * torch.randn(input_size - 1, self.n_freqs))  # to converts inputs to m_freqs
            self.B = sigma_xy * torch.randn(dim_in - 1, self.n_freqs)  # to converts inputs to m_freqs
            self.B = torch.cat([self.B, sigma_t * torch.randn(1, self.n_freqs)], dim=0)
            self.B = nn.Parameter(self.B)
            dim_in = num_neurons

        self.first = nn.Linear(dim_in, num_neurons)
        self.last = nn.Linear(num_neurons, dim_out)

        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(num_neurons, num_neurons)
                           for _ in range(num_layers_per_block)])
            for _ in range(num_resnet_blocks)])

    def forward(self, x):
        if self.fourier_features:
            # cosx = torch.cos(torch.matmul(x[..., :2], self.B))
            # sinx = torch.sin(torch.matmul(x[..., :2], self.B))
            # x = torch.cat((cosx, sinx), dim=1)
            # concatenate with time
            # x = torch.cat((x, x[..., 2:]), dim=1)
            xy_feat = two_dimensional_fourier_features(x[..., :2], self.B[:2])
            # concatenate with time
            t_feat = one_dimensional_fourier_features(x[:, :2], self.B[:2])
            x = torch.cat((xy_feat, t_feat), dim=1)
            x = self.activation(self.beta0 * self.first(x))

        else:
            x = self.activation(self.beta0 * self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(self.beta[i][0] * self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j] * self.resblocks[i][j](z))

            x = z + x

        out = self.last(x)

        return out

    def model_capacity(self):
        """
        Prints the number of parameters and the number of layers in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print("\n\nThe number of layers in the model: %d" % num_layers)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)


def get_activation_function(activation='tanh'):
    if activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()


class MLPWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=3, activation='identity',
                 siren=False, positional_encoding=False, n_freqs=5, sigma_xy=32, sigma_t=4092):
        super(MLPWithAttention, self).__init__()
        self.fourier_features = positional_encoding
        self.siren = siren
        # self.n_freqs = n_freqs
        if self.fourier_features:
            self.num_components = 80
            t_feature_size = 2 * (self.num_components + 1) - 1
            xy_feature_size = 4 * (self.num_components)
            total_feature_size = 3 * t_feature_size  # + xy_feature_size
            # self.first = nn.Linear(total_feature_size, hidden_size)
            # self.first = Siren(total_feature_size, hidden_size)
            # self.first = nn.Linear(6 * self.n_freqs, hidden_size)
            # self.B = sigma_xy * torch.randn(input_size - 1, self.n_freqs)  # to converts inputs to m_freqs
            # self.B = torch.cat([self.B, sigma_t * torch.randn(1, self.n_freqs)], dim=0)
            # self.B = nn.Parameter(self.B)
            input_size = total_feature_size

        # Initialize encoders
        if siren:  # dim_in, dim_out, w0 = 30., c = 6., is_first = False, use_bias = True, activation = None
            self.U_encoder = Siren(input_size, hidden_size, is_first=True, w0=15., c=6.)
            self.V_encoder = Siren(input_size, hidden_size, is_first=True, w0=15., c=6.)
        else:
            self.U_encoder = nn.Linear(input_size, hidden_size)
            self.V_encoder = nn.Linear(input_size, hidden_size)
        # activation
        if not siren:
            # self.activation = nn.Identity()
            self.activation = get_activation_function(activation)
        else:
            self.activation = nn.Identity()
        # Initialize MLP layers
        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            if i == 0:
                hidden_size_0 = input_size
            else:
                hidden_size_0 = hidden_size
            if siren:
                self.layers.append(Siren(hidden_size_0, hidden_size, w0=30., c=6.))
            else:
                self.layers.append(nn.Linear(hidden_size_0, hidden_size))
        if siren:
            self.output_layer = Siren(hidden_size, output_size, w0=30., c=6., activation=nn.Identity())
        else:
            self.output_layer = nn.Linear(hidden_size, output_size)
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)
            # xavier initialization of encoders
            nn.init.xavier_uniform_(self.U_encoder.weight)
            nn.init.zeros_(self.U_encoder.bias)
            nn.init.xavier_uniform_(self.V_encoder.weight)
            nn.init.zeros_(self.V_encoder.bias)

    def forward(self, x):
        if self.fourier_features:
            # xy_feat = two_dimensional_fourier_features(x[..., :2], self.B[:2])
            # # concatenate with time
            # t_feat = one_dimensional_fourier_features(x[:, :2], self.B[:2])
            x_feat = one_dimensional_fourier_features_period(x[..., 0:1], period=1 / 80,
                                                             num_components=self.num_components)
            y_feat = one_dimensional_fourier_features_period(x[..., 1:2], period=1 / 80,
                                                             num_components=self.num_components)
            t_feat = one_dimensional_fourier_features_period(x[:, 2:], period=1 / 80,
                                                             num_components=self.num_components)
            x = torch.cat((x_feat, y_feat, t_feat), dim=1)
            # x = self.activation(self.first(x))
        # Encoding step
        U = self.activation(self.U_encoder(x))  # encoder 1
        V = self.activation(self.V_encoder(x))  # encoder 2

        # MLP layers with attention
        alpha = x
        for layer in self.layers:
            alpha = layer(alpha)
            alpha = (1 - alpha) * U + alpha * V
            if not self.siren:
                alpha = self.activation(alpha)

        # Output layer
        output = self.output_layer(alpha)

        return output


def two_dimensional_fourier_features(X, b):
    # Compute cosine and sine components
    cos_x = torch.cos(b[0] * X[..., :1])  # Shape: (batch_size, num_components_x, 1)
    cos_y = torch.cos(b[1] * X[..., 1:2])  # Shape: (batch_size, 1, num_components_y)
    sin_x = torch.sin(b[0] * X[..., :1])  # Shape: (batch_size, num_components_x, 1)
    sin_y = torch.sin(b[1] * X[..., 1:2])  # Shape: (batch_size, 1, num_components_y)
    # Combine cosine and sine components
    xy_features = torch.cat([cos_x * cos_y,
                             cos_x * sin_y,
                             sin_x * cos_y,
                             sin_x * sin_y],
                            dim=-1)
    return xy_features


def one_dimensional_fourier_features(x, b_t):
    cosx = torch.cos(torch.matmul(x, b_t))
    sinx = torch.sin(torch.matmul(x, b_t))
    x_features = torch.cat((cosx, sinx), dim=1)
    return x_features


def two_dimensional_fourier_features_period(X, period, num_components=5):
    omega_x = 2 * np.pi / period
    omega_y = 2 * np.pi / period

    # Generate frequency indices
    i = torch.arange(1, num_components + 1, dtype=X.dtype, device=X.device).unsqueeze(0)  # Shape: (1, num_components_x)
    j = torch.arange(1, num_components + 1, dtype=X.dtype, device=X.device).unsqueeze(0)  # Shape: (1, num_components_y)

    # Compute cosine and sine components
    cos_x = torch.cos(omega_x * X[..., :1] * i)  # Shape: (batch_size, num_components_x, 1)
    cos_y = torch.cos(omega_y * X[..., 1:2] * j)  # Shape: (batch_size, 1, num_components_y)
    sin_x = torch.sin(omega_x * X[..., :1] * i)  # Shape: (batch_size, num_components_x, 1)
    sin_y = torch.sin(omega_y * X[..., 1:2] * j)  # Shape: (batch_size, 1, num_components_y)

    # Combine cosine and sine components
    xy_features = torch.cat(
        [
            cos_x * cos_y,
            cos_x * sin_y,
            sin_x * cos_y,
            sin_x * sin_y
        ],
        dim=-1
    )  # Shape: (batch_size, num_components_x, num_components_y, 4)

    return xy_features


def one_dimensional_fourier_features_period(X, period, num_components=5):
    omega_x = 2 * np.pi / period
    i = torch.arange(1, num_components + 1, dtype=X.dtype, device=X.device)
    x_features = torch.cat([torch.ones_like(X), torch.cos(i * omega_x * X), torch.sin(i * omega_x * X)], dim=1)
    return x_features
