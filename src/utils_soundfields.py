import numpy as np
import matplotlib.pyplot as plt
# from numba import jit
import matplotlib as mpl
from scipy.interpolate import griddata
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import linear_model
from sklearn.neighbors import KernelDensity,NearestNeighbors
from sklearn.cluster import MeanShift
from matplotlib import transforms
from matplotlib import cm
from matplotlib.colors import ListedColormap
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
def find_MAP( x):
    try:
        mean_shift = MeanShift()
        mean_shift.fit(x)
        centers = mean_shift.cluster_centers_
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x)

        best_center = (None, -np.inf)
        dens = kde.score_samples(centers)
        for c, d in zip(centers, dens):
            if d > best_center[1]:
                best_center = (c.copy(), d)

        dist_to_best = np.sum((x - best_center[0]) ** 2, axis=1)
        return np.argmin(dist_to_best)
    except:
        print('Mean shift failed')
        return 0


# @jit(nopython=True)
def stack_real_imag_H(mat):
    mat_stack = np.concatenate(
        (
            np.concatenate((mat.real, -mat.imag), axis=-1),
            np.concatenate((mat.imag, mat.real), axis=-1),
        ),
        axis=0,
    )
    return mat_stack

def plot_array_pressure(p_array, array_grid, ax=None, plane = False, norm = None, z_label = False,
                        cmp = None):
    if ax is None:
        if z_label:
            ax = plt.axes(projection='3d')
        else:
            ax = plt.axes()
    if cmp is None:
        cmp = plt.get_cmap("RdBu")
    else:
        cmp = plt.get_cmap(cmp)
    if norm is None:
        vmin = p_array.real.min()
        vmax = p_array.real.max()
    else:
        vmin, vmax = norm
    if z_label:
        sc = ax.scatter(array_grid[0], array_grid[1], array_grid[2], c=p_array.real,
                        cmap=cmp, alpha=1., s=10, vmin = vmin, vmax = vmax)
    else:
        sc = ax.scatter(array_grid[0], array_grid[1], c=p_array.real,
                        cmap=cmp, alpha=1., s=10, vmin = vmin, vmax = vmax)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    if z_label:
        ax.set_zlabel('z [m]')
        ax.view_init(45, 45)

        if plane:
            ax.set_box_aspect((1,1,1))
        else:
            ax.set_box_aspect((array_grid[0].max(), array_grid[1].max(), array_grid[2].max()))
    return ax, sc

# @jit(nopython=True)
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

# @jit(nopython=True)
def _disk_grid_fibonacci(n, r, c = (0,0), z=None):
    """
    Get circular disk grid points
    Parameters
    ----------
    n : integer N, the number of points desired.
    r : float R, the radius of the disk.
    c : tuple of floats C(2), the coordinates of the center of the disk.
    z : float (optional), height of disk
    Returns
    -------
    cg :  real CG(2,N) or CG(3,N) if z != None, the grid points.
    """
    r0 = r / np.sqrt(float(n) - 0.5)
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    gr = np.zeros(n)
    gt = np.zeros(n)
    for i in range(0, n):
        gr[i] = r0 * np.sqrt(i + 0.5)
        gt[i] = 2.0 * np.pi * float(i + 1) / phi

    if z is None:
        cg = np.zeros((3, n))
    else:
        cg = np.zeros((2, n))

    for i in range(0, n):
        cg[0, i] = c[0] + gr[i] * np.cos(gt[i])
        cg[1, i] = c[1] + gr[i] * np.sin(gt[i])
        if z != None:
            cg[2, i] = z
    return cg

# @jit(nopython=True)
def propagation_matmul(H, x):
    # return np.einsum('ijk, ik -> ij', H, x)
    return H @ x

def fib_sphere(num_points, radius=1.):
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

    x_batch = np.dot(radius, x)
    y_batch = np.dot(radius, y)
    z_batch = np.dot(radius, z)

    # Display points in a scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_batch, y_batch, z_batch, s = 3)
    # plt.show()
    return np.asarray([x_batch, y_batch, z_batch])

def wavenumber(f=1000, c = 343):
    omega = 2 * np.pi * f  # angular frequency
    k = omega / c  # wavenumber
    return k

def SNRScale(sig, snrdB = 40):
    ndim = sig.ndim
    if ndim >= 2:
        dims = (-2, -1)
    else:
        dims = -1
    mean = np.mean(sig, axis=dims)
    # remove DC
    if ndim > 2:
        sig_zero_mean = sig - mean[..., np.newaxis, np.newaxis]
    else:
        sig_zero_mean = sig - mean[..., np.newaxis]

    var = np.var(sig_zero_mean, axis=dims)
    if ndim >= 2:
        psig = var[..., np.newaxis, np.newaxis]
    else:
        psig = var[..., np.newaxis]

    # For x dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    snr_lin = 10.0 ** (snrdB / 10.0)

    # Find required noise power
    return psig / snr_lin

def adjustSNR(sig, snrdB=40, td=True):
    """
    Add zero-mean, Gaussian, additive noise for specific SNR
    to input signal

    Parameters
    ----------
    sig : Tensor
        Original Signal.
    snrdB : int, optional
        Signal to Noise ratio. The default is 40.

    Returns
    -------
    x : Tensor
        Noisy Signal.

    """
    # Signal power in data from signal
    ndim = sig.ndim
    if ndim >= 2:
        dims = (-2, -1)
    else:
        dims = -1
    mean = np.mean(sig, axis=dims)
    # remove DC
    if ndim > 2:
        sig_zero_mean = sig - mean[..., np.newaxis, np.newaxis]
    else:
        sig_zero_mean = sig - mean[..., np.newaxis]

    var = np.var(sig_zero_mean, axis=dims)
    if ndim >= 2:
        psig = var[..., np.newaxis, np.newaxis]
    else:
        psig = var[..., np.newaxis]

    # For x dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    snr_lin = 10.0 ** (snrdB / 10.0)

    # Find required noise power
    pnoise = psig / snr_lin

    if td:
        # Create noise vector
        noise = np.sqrt(pnoise) * np.random.randn(sig.shape)
    else:
        # complex valued white noise
        real_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        imag_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        noise = real_noise + 1j * imag_noise
        noise_mag = np.sqrt(pnoise) * np.abs(noise)
        noise = noise_mag * np.exp(1j * np.angle(noise))

    # Add noise to signal
    sig_plus_noise = sig + noise
    return sig_plus_noise

def get_spherical_array(n_mics, radius, add_interior_points = True, reference_grid = None):
    if not add_interior_points:
        return fib_sphere(n_mics, radius)
    else:
        assert reference_grid is not None
        rng = np.random.RandomState(1234)
        grid = fib_sphere(n_mics, radius)
        npoints = 5
        x_ref, y_ref, z_ref = reference_grid
        # number of interior points for zero-cross of bessel functions
        mask = np.argwhere(x_ref.ravel() ** 2 + y_ref.ravel() ** 2 <= radius ** 2)
        interp_ind = rng.choice(mask.shape[0], size=npoints, replace=False)
        interp_ind = np.squeeze(mask[interp_ind])
        grid = np.concatenate((grid, reference_grid[:, interp_ind]), axis=-1)
        return grid

def FindInterpolationIndex(grid_ref, grid):
    rng = np.random.RandomState(1234)
    npoints = 5
    mu = grid.mean(axis=-1)[..., None]
    tempgrid = grid - mu
    radius = np.linalg.norm(tempgrid, axis = 0).min()
    mask = np.argwhere(grid_ref[0].ravel() ** 2 + grid_ref[1].ravel() ** 2 <= radius ** 2)
    index = rng.choice(mask.shape[0], size=npoints, replace=False)
    index = np.squeeze(mask[index])
    return index

def FindExtrapolationIndex(grid_ref, grid):
    rng = np.random.RandomState(1234)
    npoints = 5
    mu = grid.mean(axis=-1)[..., None]
    tempgrid = grid - mu
    radius = np.linalg.norm(tempgrid, axis = 0).min()
    mask = np.argwhere(grid_ref[0].ravel() ** 2 + grid_ref[1].ravel() ** 2 > radius ** 2)
    index = rng.choice(mask.shape[0], size=npoints, replace=False)
    index = np.squeeze(mask[index])
    return index

def ConcatMeasurements(data1, data2, concatindex):
    if data1.ndim >1:
        m,n = data1.shape
    else:
        m = len(data1)
        n = 1
    if m < n:
        data = np.concatenate((data1, data2[:, concatindex]), axis=-1)
    else:
        data = np.concatenate((data1, data2[concatindex]), axis=-1)
    return data

# grid = grids_sphere - grids_sphere.mean(axis = -1)[..., None]
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(grid[0],grid[1],grid[2])
# fig.show()
def distance_between(s, r):
    """Distance of all combinations of points in s and r.
    Parameters
    ----------
    s : ndarray, (3, ns)
    r : ndarray, (3, nr)
    Returns
    -------
    ndarray, (nr, ns)
        Distances between points
    """
    return np.linalg.norm(s[:, None, :] - r[:, :, None], axis=0)

def plane_waves(n0, k, grid, orthonormal = True):
    """
    x0 : (3,) array_like
        Position of plane wave.
    n0 : (3,) array_like
        Normal vector (direction) of plane wave.
    grid : triple of array_like
        The grid that is used for the sound field calculations.
    """

    # n0 = n0 / np.linalg.norm(n0, axis = -1)[..., None]

    P = np.exp(-1j * k * grid.T@n0)
    if orthonormal:
        P /= np.sqrt(n0.shape[-1])
    return P

def get_random_np_boolean_mask( n_true_elements, total_n_elements):
        assert total_n_elements >= n_true_elements
        a = np.zeros(total_n_elements, dtype=int)
        a[:n_true_elements] = 1
        np.random.shuffle(a)
        return a.astype(bool)

def plot_pixel_sound_field(p, x, y, cmap='coolwarm', clim=None, ax = None):
    """
    Plot sound field with higher resolution grid points
    ----------------------------------------
    Args:
        p : Pressure vector
        x : X grid points
        y : Y grid points
        x_hi : X grid points with higher resolution
        y_hi : Y grid points with higher resolution
        cmap : Colormap (default: 'coolwarm')
        clim : Colorbar limits (default: None)
    """
    upsample_factor = 1
    steps =  int(upsample_factor*np.ceil(np.sqrt(p.shape[0])))
    res = complex(0, steps)
    X, Y = np.mgrid[x.min():x.max():res, y.min():y.max():res]
    points = np.c_[x, y]
    P = griddata(points, p, (X, Y), method='cubic', rescale=True)
    # append zeros between points of Pmesh_lo to get P_hi
    mask = np.add.outer(range(steps), range(steps)) % upsample_factor
    dx = 0.5 * X.ptp() / P.shape[1]
    dy = 0.5 * Y.ptp() / P.shape[0]
    if ax is None:
        _, ax = plt.subplots()
    degrees = -270
    tr = transforms.Affine2D().rotate_deg(degrees)
    im = ax.imshow(P, cmap=cmap, origin='upper',
                   extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy],
                   transform=tr + ax.transData)
    # overlay zeros
    # _ = ax.imshow(mask, origin='upper',
    #               extent=[X_hi.min() - dx, X_hi.max() + dx, Y_hi.min() - dy, Y_hi.max() + dy],
    #               transform=tr + ax.transData, cmap='Greys', alpha=0.4)
    # ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    if clim is not None:
        im.set_clim(clim)
    # plt.colorbar(im, ax=ax)
    # plt.show()
    return ax, im
def plot_sf(P, x, y, f=None, ax=None, name=None, save=False, add_meas=None,
            clim=None, tex=False, cmap=None, normalise=False,
            colorbar = False, cbar_label = '', cbar_loc = 'bottom',
            interpolated = True, markersize = 10, pixels = True,
            N_interp = 256, dilation = False):
    """
    Plot spatial soundfield normalised amplitude
    --------------------------------------------
    Args:
        P : Pressure in meshgrid [X,Y]
        X : X mesh matrix
        Y : Y mesh matrix
    Returns:
        ax : pyplot axes (optionally)
    """
    # plot_settings()


    if normalise:
        Pvec = P / np.max(abs(P))
    else:
        Pvec = P
    res = complex(0, N_interp)
    Xc, Yc = np.mgrid[x.min():x.max():res, y.min():y.max():res]
    points = np.c_[x, y]
    Pmesh = griddata(points, Pvec, (Xc, Yc), method='cubic', rescale=True)
    upsample_grid = np.c_[Xc.flatten(), Yc.flatten()]
    if dilation:
        # create empty vector with dilation grid points
        P_hi = np.zeros(Pmesh.shape)
        _, indices = subsample_gridpoints(upsample_grid.T, comparison_grid= points.T)
        # plt.scatter(upsample_grid[indices, 0], upsample_grid[indices, 1], c='k')
        P_hi = P_hi.flatten()
        P_hi[indices] = Pvec
        mask = np.zeros_like(P_hi)  # Creating a mask of zeros with the same shape as Pmesh.real
        mask[indices] = 1  # Set the values at the specified indices to 1 to create the mask
        Pmesh = P_hi.reshape(Pmesh.shape)
        # mask image with dilation grid at indices
        mask = mask.reshape(Pmesh.shape)
    else:
        mask = np.ones(Pmesh.shape)
    if cmap is None:
        cmap = 'coolwarm'
    if f is None:
        f = ''
    # P = P / np.max(abs(P))
    X = Xc.flatten()
    Y = Yc.flatten()
    if tex:
        plt.rc('text', usetex=True)
    # x, y = X, Y
    # clim = (abs(P).min(), abs(P).max())
    dx = 0.5 * X.ptp() / Pmesh.size
    dy = 0.5 * Y.ptp() / Pmesh.size
    if ax is None:
        _, ax = plt.subplots()  # create figure and axes
    if interpolated:
        degrees = -270
        tr = transforms.Affine2D().rotate_deg(degrees)
        alpha_mask = 1 - mask  # Inverting the mask (0 becomes 1 and 1 becomes 0)
        alpha = np.where(alpha_mask, 0.,
                         1.)  # Set alpha values to 1. where the mask is 0, and 0 where the mask is 1

        im = ax.imshow(Pmesh.real, cmap=cmap, origin='upper', alpha = alpha,
                       extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy],
                       transform =tr + ax.transData)
        # if dilation:
        #     # get custom colormap for mask (black and white)
        #     cmap = cm.get_cmap('Greys', 2)
        #     cmap.set_bad(color='white')
        #     alpha_mask = 1 - mask  # Inverting the mask (0 becomes 1 and 1 becomes 0)
        #     alpha = np.where(alpha_mask, 0.,
        #                      0)  # Set alpha values to 0.5 where the mask is 0, and 0 where the mask is 1
        #     # Overlaying the mask on the image
        #     ax.imshow(mask, cmap='Greys', alpha=alpha, extent=[X.min() - dx, X.max() + dx, Y.min() - dy, Y.max() + dy],
        #               origin='upper', transform=tr + ax.transData)

        # ax.invert_xaxis()
    elif pixels:
        if clim is not None:
            lm1, lm2 = clim
        else:
            lm1, lm2 = None, None
        ax, im  = plot_pixel_sound_field(Pvec.real, x,y, cmap=cmap, ax = ax)
    else:
        if clim is not None:
            lm1, lm2 = clim
        else:
            lm1, lm2 = None, None
        im = ax.scatter(x, y, c=Pvec.real,
                        cmap=cmap, alpha=1., s=markersize, vmin = lm1, vmax = lm2)
        ax.set_aspect('equal')
    ax.set_ylabel('y (m)')
    ax.set_xlabel('x (m)')
    if clim is not None:
        lm1, lm2 = clim
        im.set_clim(lm1, lm2)
    if colorbar:
        if cbar_loc != 'bottom':
            shrink = 1.
            orientation = 'vertical'
        else:
            shrink = 1.
            orientation = 'horizontal'

        cbar = plt.colorbar(im, ax = ax, location=cbar_loc,
                            shrink=shrink)
        # cbar.ax.get_yaxis().labelpad = 15
        titlesize = mpl.rcParams['axes.titlesize']
        # cbar.ax.set_title(cbar_label, fontsize = titlesize)
        cbar.set_label(cbar_label, fontsize = titlesize)
    if add_meas is not None:
        x_meas = X.ravel()[add_meas]
        y_meas = Y.ravel()[add_meas]
        ax.scatter(x_meas, y_meas, s=1, c='k', alpha=0.3)

    if name is not None:
        ax.set_title(name)
    if save:
        plt.savefig(name + '_plot.png', dpi=150)
    return ax, im

def plot_sf_quiver(u, x, y, ax=None,tex=False, colormax = None, colormap = 'Greys'):
    """
    Plot sound field quantities such as velocity or intensity using quiver
    ---------------------------------------------------------------------
    Args:
        P : quantity to plot in meshgrid
        X : X mesh matrix
        Y : Y mesh matrix
    Returns:
        ax : pyplot axes (optionally)

    """
    if tex:
        plt.rc('text', usetex=True)
    # plot quiver
    if ax is None:
        _, ax = plt.subplots()
    dx, dy = u[0, :], u[1, :]
    # grad = np.gradient(u, axis=-1)
    # dx, dy = grad[0, :], grad[1, :]
    if colormax is None:
        M = np.hypot(dx, dy)
        # M /= np.linalg.norm(M)
        # n = 0
        # M = np.sqrt(dx ** 2 + dy ** 2)
    else:
        M = colormax
    # q = ax.quiver(x, y, dx, dy, M,  width=0.003, headwidth=5, headlength=10, headaxislength=8)
    q = ax.quiver(x, y, dx, dy, M, units = 'xy', width=0.003, headwidth=7, headlength=10, headaxislength=8,
                  cmap = colormap, pivot = 'mid', alpha = 0.9)
    # ax, q = plot_cones(ax, x, y, -dx, -dy, scale=50, angle=20, cmap='Greys', intensity = M)
    # q = ax.streamplot(x, y, dx, dy, density=1)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    return ax, q

def plot_cones(ax, x, y, u, v, intensity, scale=1, angle=30, cmap='viridis', **kwargs):
    """
    Plots cones instead of arrowheads on a 2D quiver plot, with colors based on an intensity map.

    Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): The Axes object to draw on.
        x (array-like): The x-coordinates of the arrow locations.
        y (array-like): The y-coordinates of the arrow locations.
        u (array-like): The x-components of the arrow vectors.
        v (array-like): The y-components of the arrow vectors.
        intensity (array-like): Intensity values corresponding to each arrow.
        scale (float, optional): The arrow scaling. Default is 1.
        angle (float, optional): The angle of the cone apex in degrees. Default is 30.
        cmap (str, optional): The colormap to use for converting intensity to colors. Default is 'viridis'.
        **kwargs: Other keyword arguments to pass to the fill function.

    Returns:
        list: The list of cones.
    """
    # Compute the cone angles and lengths
    angles = np.arctan2(v, u)
    lengths = np.sqrt(u**2 + v**2) * scale

    # Convert intensity values to colors using the specified colormap
    norm = plt.Normalize(vmin=np.min(intensity), vmax=np.max(intensity))
    colors = plt.get_cmap(cmap)(norm(intensity))

    cones = []
    for xi, yi, angle_rad, length, color in zip(x, y, angles, lengths, colors):
        cone = ax.fill(
            [
                xi,
                xi + length * np.cos(angle_rad),
                xi + length * np.cos(angle_rad + np.deg2rad(angle)),
            ],
            [
                yi,
                yi + length * np.sin(angle_rad),
                yi + length * np.sin(angle_rad + np.deg2rad(angle)),
            ],
            color=color,
            **kwargs
        )
        cones.extend(cone)

    # get colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax)


    return ax, sm

# # # Example usage
# x = np.linspace(0, 10, 5)
# y = np.linspace(0, 10, 5)
# u = np.cos(np.linspace(0, np.pi, 5))
# v = np.sin(np.linspace(0, np.pi, 5))
# intensity = np.linspace(0, 1, 5)  # Example intensity values from 0 to 1
#
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
#
# plot_cones(ax, x, y, u, v, intensity, scale=1, angle=30, cmap='plasma')
#
# plt.show()

def plot_rir(rir, title='', t_intervals=None, ax = None):
    if t_intervals is None:
        t_intervals = [.01, .2]
    width = 6.694
    normalise_response = lambda x : x/np.max(abs(x))
    # width = plot_settings()
    # fig, ax = plt.subplots(1, 1, figsize=(width, width/4))

    rir = normalise_response(rir)
    fs = 16000
    t = np.linspace(0, len(rir) / fs, len(rir))
    t_ind = np.argwhere((t > t_intervals[0]) & (t < t_intervals[1]))[:, 0]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(width, width/4))
    ax.plot(t[t_ind], rir[t_ind], linewidth=3, color='seagreen')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Normalised SP [Pa]')
    ax.set_ylim([np.min(rir) - .1 * np.max(rir), np.max(rir) + .1 * np.max(rir)])
    ax.grid(linestyle=':', which='both', color='k')
    ax.set_title(title)
    return ax


def plot_settings():
    width = 6.694

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    }

    mpl.rcParams.update(tex_fonts)
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    # plt.rcParams["figure.figsize"] = (6.694, 5)
    plt.rcParams['figure.constrained_layout.use'] = True
    return width


def array_to_cmplx(array):
    return array[..., 0] + 1j * array[..., 1]

def sample_unit_sphere(n_samples=2000):
    grid = fib_sphere(int(n_samples), 1)
    return grid
# a = 0
# bs = [0.0002]
# for b in bs:
#     x = np.linspace(stats.halfcauchy.ppf(0.0001, a, b),
#                     stats.halfcauchy.ppf(0.9999, a, b), 100)
#     plt.plot(x, stats.halfcauchy.pdf(x, a, b),
#              lw=5, alpha=0.6, label= f'b = {b}')
# plt.legend()
# plt.show()

# for ii in range (10000):
#     b_log_r = np.random.normal(4,1)
#     b_r = 10**(-b_log_r)
#     b_log_im= np.random.normal(3.5,1)
#     b_im = 10**(-b_log_im)
#     mu_r = 0.
#     # tau_r = np.random.lognormal(0., 1., size=shape)
#     tau_r = stats.invgamma.rvs(8, b_r, size=shape)
#     mu_im = 0.
#     tau_im = stats.invgamma.rvs(8, b_im, size=shape)
#     if np.logical_or(tau_im.max()>2, tau_r.max()>2):
#         print(f"tau im max: {tau_im.max()}, min: {tau_im.min()}")
#         print(f"tau re max: {tau_r.max()}, min: {tau_r.min()}")
def hierarchical_prior(shape=(1, 900), data_scale = 1.):
    
    b_log_r = np.random.normal(2,1)
    b_r = 10**(-b_log_r)
    b_log_im= np.random.normal(2,1)
    b_im = 10**(-b_log_im)

    mu_r = 0.
    tau_r = stats.invgamma.rvs(3, b_r, size=shape)
    mu_im = 0.
    tau_im = stats.invgamma.rvs(3, b_im, size=shape)


    coeffs_r = np.random.normal(mu_r, data_scale*tau_r, size=shape)
    coeffs_i = np.random.normal(mu_im, data_scale*tau_im, size=shape)

    return coeffs_r + 1j*coeffs_i

def GaussianLikelihood(Hm, priorcoeff, snrdB_range = [15, 35]):
    mu = propagation_matmul(Hm, priorcoeff)
    dim = mu.shape[0]
    SNR = np.random.uniform(snrdB_range[0], snrdB_range[1], size = mu.shape)
    scale = SNRScale(mu, SNR)
    # iid entries for real and imag (also circularly symmetric)
    yreal = np.random.normal(mu.real, scale)
    yimag = np.random.normal(mu.imag, scale)
    # Circularly-symmetric - works fine
    # ymu = np.concatenate((mu.real, mu.imag))
    # yscale_re = scale*np.eye(mu.shape[0])
    # yscale_im = yscale_re.copy()
    # yscale1 = np.concatenate((yscale_re, -yscale_im), axis= -1)
    # yscale2 = np.concatenate((yscale_im, yscale_re), axis= -1)
    # yscale = np.concatenate((yscale1, yscale2), axis = 0)
    # y = np.random.multivariate_normal(ymu, yscale)
    # y_complex = y[:dim] + 1j*y[dim:]
    # y = np.random.normal(mu, scale)
    y_complex = yreal + 1j*yimag
    return y_complex

class PlaneWaveDataset(Dataset):
    def __init__(self, n_soundfields, n_mics, freq, temperature, n_plwaves=900,
                 snr=None, hierarchical_bayes = False, mic_array_grid = None,
                 reference_grid = None,
                 n_active_waves = np.inf, single_net = True,
                 sparse = False, priorstd = None):
        self.n_mics = n_mics
        self.freq = freq
        self.n_soundfields = n_soundfields
        self.temperature = temperature
        self.n_plwaves = n_plwaves
        self.snr = snr
        self.sparse = sparse
        self.hierarchical_bayes = hierarchical_bayes
        self.n_active_waves = n_active_waves
        self.speed_of_sound = speed_of_sound(self.temperature)
        self.wavenumber = wavenumber(self.freq, self.speed_of_sound)
        if reference_grid is None:
            self.grid_ref = _disk_grid_fibonacci(n = 800, r = 1.4)
        else:
            self.grid_ref = reference_grid

        if priorstd is None:
            self.priorstd = 5
        else:
            self.priorstd = priorstd

        if mic_array_grid is None:
            self.mic_array_grid = get_spherical_array(self.n_mics,
                                                      radius= 1.2,
                                                      reference_grid=self.grid_ref)
        else:
            self.mic_array_grid = mic_array_grid
            self.n_mics = self.mic_array_grid.shape[-1]
        self.sample_directions = sample_unit_sphere(self.n_plwaves)
        Hm = plane_waves(self.sample_directions, self.wavenumber, self.mic_array_grid)
        self.Hm = Hm.astype(np.complex64)
        self.Href = plane_waves(self.sample_directions, self.wavenumber, self.grid_ref)
        self.single_net = single_net
        # print("CHECK CHECK CHECK")
    def __getitem__(self, index):

        if self.sparse:
            coeffs = np.random.laplace(0, self.priorstd *np.sqrt(2)/2, (self.n_plwaves)) + 1j*np.random.laplace(0, self.priorstd *np.sqrt(
                2)/2, (self.n_plwaves))
        else:
            # coeffs = np.random.normal(0, self.priorstd * np.sqrt(2) / 2, (self.n_plwaves)) + 1j * np.random.normal(0, self.priorstd * np.sqrt(
            #     2) / 2, (self.n_plwaves))
            # coeffs = np.random.normal(0, self.priorstd, (self.n_plwaves)) + 1j * np.random.normal(0, self.priorstd, (self.n_plwaves))
            coeffs = hierarchical_prior(shape =  (self.n_plwaves,))
        activations = np.ones_like(coeffs).astype(np.complex64)
        # activations = np.random.binomial(n = 1, p = 0.002, size= self.n_plwaves).astype(np.complex64)
        coeffs = activations*coeffs.astype(np.complex64)
        x = torch.view_as_real(torch.from_numpy(coeffs))


        # p = propagation_matmul( self.Hm, coeffs)
        # if self.snr is not None:
        #     p = adjustSNR(p, snrdB = self.snr, td = False).astype(np.complex64)
        p = GaussianLikelihood(self.Hm, coeffs).astype(np.complex64)
        y = torch.view_as_real(torch.from_numpy(p))

        if self.single_net:
            return torch.cat((x[..., 0], x[..., 1])), torch.cat((y[..., 0], y[..., 1]))
        else:
            return x, y

    def __len__(self):
        return self.n_soundfields

class PlaneWaveDatasetBroadband(Dataset):
    def __init__(self, n_soundfields, n_mics,  temperature, n_plwaves=900,
                 snr=None, hierarchical_bayes = False, mic_array_grid = None,
                 reference_grid = None,
                 n_active_waves = np.inf, single_net = True,
                 sparse = False, priorstd = None,
                 freq_vector = np.fft.rfftfreq(8192, 1/8000),
                 uniform_prior = False, normal_prior = False):
        self.n_mics = n_mics
        self.freq_vector = freq_vector
        self.n_soundfields = n_soundfields
        self.temperature = temperature
        self.n_plwaves = n_plwaves
        self.snr = snr
        self.sparse = sparse
        self.uniform_prior = uniform_prior
        self.normal_prior = normal_prior
        self.hierarchical_bayes = hierarchical_bayes
        self.n_active_waves = n_active_waves
        self.speed_of_sound = speed_of_sound(self.temperature)
        self.wavenumber = wavenumber(self.freq_vector, self.speed_of_sound)
        if reference_grid is None:
            self.grid_ref = _disk_grid_fibonacci(n = 800, r = 1.4)
        else:
            self.grid_ref = reference_grid

        if priorstd is None:
            self.priorstd = 1
        else:
            self.priorstd = priorstd

        if mic_array_grid is None:
            self.mic_array_grid = get_spherical_array(self.n_mics,
                                                      radius= 1.2,
                                                      reference_grid=self.grid_ref)
        else:
            self.mic_array_grid = mic_array_grid
            self.n_mics = self.mic_array_grid.shape[-1]
        self.sample_directions = sample_unit_sphere(self.n_plwaves)
        # Hm = plane_waves(self.sample_directions, self.wavenumber, self.mic_array_grid)
        # self.Hm = Hm.astype(np.complex64)
        # self.Href = plane_waves(self.sample_directions, self.wavenumber, self.grid_ref)
        self.single_net = single_net
        # print("CHECK CHECK CHECK")
    def __getitem__(self, index):
        Hm = plane_waves(self.sample_directions, self.wavenumber[index], self.mic_array_grid)
        if self.sparse:
            coeffs = np.random.laplace(0, self.priorstd *np.sqrt(2)/2, (self.n_plwaves)) + 1j*np.random.laplace(0, self.priorstd *np.sqrt(
                2)/2, (self.n_plwaves))
        elif self.uniform_prior:
            min, max = -1., 1.
            coeffs = np.random.uniform(min,max, (self.n_plwaves)) + 1j* np.random.uniform(min,max, (self.n_plwaves))
        elif self.normal_prior:
            coeffs = np.random.normal(0, self.priorstd *np.sqrt(2)/2, (self.n_plwaves)) + 1j*np.random.normal(0, self.priorstd *np.sqrt(
                2)/2, (self.n_plwaves))
        else:
            # coeffs = np.random.normal(0, self.priorstd * np.sqrt(2) / 2, (self.n_plwaves)) + 1j * np.random.normal(0, self.priorstd * np.sqrt(
            #     2) / 2, (self.n_plwaves))
            coeffs = hierarchical_prior(shape =  (self.n_plwaves,), data_scale= self.priorstd)

        coeffs = coeffs.astype(np.complex64)
        x = torch.view_as_real(torch.from_numpy(coeffs))


        # p = propagation_matmul( self.Hm, coeffs)
        # if self.snr is not None:
        #     p = adjustSNR(p, snrdB = self.snr, td = False).astype(np.complex64)
        p = GaussianLikelihood(Hm, coeffs).astype(np.complex64)
        y = torch.view_as_real(torch.from_numpy(p))

        if self.single_net:
            return torch.cat((x[..., 0], x[..., 1])), torch.cat((y[..., 0], y[..., 1]))
        else:
            return x, y

    def __len__(self):
        return self.n_soundfields

def Ridge_regression(H, p, n_plwav=None, cv=True):
    """
    Titkhonov - Ridge regression for Soundfield Reconstruction
    Parameters
    ----------
    H : Transfer mat.
    p : Measured pressure.
    n_plwav : number of plane waves.

    Returns
    -------
    q : Plane wave coeffs.
    alpha_titk : Regularizor
    """
    if cv:
        reg = linear_model.RidgeCV(cv=5, alphas=np.geomspace(1, 1e-7, 50),
                                   fit_intercept=True)
    else:
        alpha_titk = 2.8e-5
        reg = linear_model.Ridge(alpha=alpha_titk, fit_intercept=True)

    # gcv_mode = 'eigen')
    # reg = linear_model.RidgeCV()
    if n_plwav is None:
        n_plwav = H.shape[-1]
    if np.logical_or(np.logical_or(H.dtype == complex, H.dtype == np.complex64), H.dtype == np.complex128):
        H = stack_real_imag_H(H)
    if np.logical_or(np.logical_or(p.dtype == complex, p.dtype == np.complex64), p.dtype == np.complex128):
        p = np.concatenate((p.real, p.imag))

    reg.fit(H, p)
    q = reg.coef_[:n_plwav] + 1j * reg.coef_[n_plwav:]
    try:
        alpha_titk = reg.alpha_
    except:
        pass
    # Predict
    return q, alpha_titk

def standardize(data):
    mu = data.mean(axis = -1)[..., None]
    sigma = data.std(axis = -1)[..., None]
    newdata =  (data - mu)/sigma
    return newdata, mu, sigma
def rescale(data, mu, sigma):
    return data*sigma + mu

def rescale_real_imag(data, mu_r, sigma_r, mu_im, sigma_im):
    rescaled_data = []
    shp = data.shape
    if not torch.is_tensor(data):
        wasnumpy = True
        data = torch.from_numpy(data)
    else:
        device = data.device
        mu_r = mu_r.to(device)
        sigma_r = sigma_r.to(device)
        mu_im = mu_im.to(device)
        sigma_im = sigma_im.to(device)
        wasnumpy = False
    if data.ndim < 2:
        data = data.unsqueeze(0)
    for d in data:
        dreal = rescale(d.real, mu_r, sigma_r)
        dimag = rescale(d.imag, mu_im, sigma_im)
        if wasnumpy:
            rescaled_data.append(dreal.numpy() + 1j*dimag.numpy())
        else:
            rescaled_data.append(dreal + 1j*dimag)
    if wasnumpy:
        return np.array(rescaled_data).reshape(shp)
    else:
        return torch.stack(rescaled_data).reshape(shp)

def scale_maxabs(data, constant = 1):
    if not torch.is_tensor(data):
        data = torch.from_numpy(data)
    maxabs = abs(data.numpy().max())
    datanew = data * constant/maxabs
    return datanew, constant/maxabs

def rescale_maxabs(data, scale):
    return data * scale
# coeffs =  hierarchical_prior(shape = (20, 1200))
#
# for i in range(20):
#     plt.hist(coeffs[i].real, bins = 300)
# plt.show()
#
def scale_linear_regression( yhat, y):
        if np.logical_or(np.logical_or(yhat.dtype == complex, yhat.dtype == np.complex64), yhat.dtype == np.complex128):
            yhat = np.concatenate((yhat.real, yhat.imag))
        if yhat.ndim < 2:
            yhat = yhat.reshape(-1, 1)
        if np.logical_or(np.logical_or(y.dtype == complex, y.dtype == np.complex64),
                         y.dtype == np.complex128):
            y = np.concatenate((y.real, y.imag))
        if y.ndim < 2:
            y = y.reshape(-1, 1)
        reg = linear_model.RidgeCV(cv=20, alphas=np.geomspace(100, 1e-2, 500),
                                   fit_intercept=False)
        reg.fit(yhat, y)
        sigma = np.squeeze(reg.coef_)
        mu = np.squeeze(reg.intercept_)
        return sigma, mu

def nmse(y_meas, y_predicted):
    num = np.sum(abs(y_meas - y_predicted)**2)
    denom = np.sum(abs(y_meas)**2)
    return 10*np.log10(num/denom)
