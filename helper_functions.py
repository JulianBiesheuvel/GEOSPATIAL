import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import numpy as np

def plot_gp_samples(X, X_syn, y, covariance_matrix, covariance_syn, description, kernel, fig, gs, xlim, xlim_cov, scatter=True):
    """
    Plot kernel matrix and samples from a Gaussian Process.

    Parameters:
    - X (numpy array): Input data.
    - X_syn (numpy array): Synthetic input data.
    - y (list of numpy arrays): List of three different function realizations.
    - covariance_matrix (numpy array): Covariance matrix.
    - covariance_syn (numpy array): Covariance matrix for synthetic data.
    - description (str): Description for the covariance matrix.
    - kernel (object): Kernel object used in the Gaussian Process Regressor.
    - fig (matplotlib.figure.Figure): Figure object for the plot.
    - gs (matplotlib.gridspec.GridSpec): GridSpec object for subplot layout.
    - xlim (tuple): Tuple representing the x-axis limits.
    - xlim_cov (tuple): Tuple representing the x-axis limits for the covariance matrix plot.
    - scatter (bool): Whether to plot scattered points in addition to the mean function.

    Returns:
    - None
    """
    gp = GPR(kernel=kernel, optimizer=None)
    f_mean, f_var = gp.predict(X, return_std=True)

    # Plot samples
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(X, f_mean, 'b-', label='Mean Function')
    upper_bound = f_mean + 1.96 * f_var
    lower_bound = f_mean - 1.96 * f_var
    ax1.fill_between(X.ravel(), lower_bound, upper_bound, color='b', alpha=0.1, label='95% Confidence Interval')

    if scatter:
        for i in range(3):
            ax1.plot(X, y[i], ':')

    ax1.set_ylabel('$y(x)$', fontsize=13, labelpad=0)
    ax1.set_xlabel('$x$', fontsize=13, labelpad=0)
    ax1.set_xlim(xlim)

    ax1.set_title(f'3 Different Function Realisations, Sampled from a Gaussian Process')
    ax1.legend(loc='lower right')

    # Plot covariance matrix
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(covariance_syn)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.02)
    cbar = plt.colorbar(im, ax=ax2, cax=cax)
    cbar.ax.set_ylabel('Covariance ($K(X,X)$)', fontsize=8)
    ax2.set_title(f'Covariance matrix\n{description}')
    ax2.set_xlabel('X', fontsize=10, labelpad=0)
    ax2.set_ylabel('X', fontsize=10, labelpad=0)

    # Show 5 custom ticks on x and y axis of covariance plot
    nb_ticks = 5
    ticks = list(range(xlim_cov[0], xlim_cov[1] + 1))
    ticks_idx = np.rint(np.linspace(1, len(ticks), num=min(nb_ticks, len(ticks))) - 1).astype(int)
    ticks = list(np.array(ticks)[ticks_idx])

    ax2.set_xticks(np.linspace(0, len(X_syn), len(ticks)))
    ax2.set_yticks(np.linspace(0, len(X_syn), len(ticks)))
    ax2.set_xticklabels(ticks)
    ax2.set_yticklabels(ticks)

    ax2.grid(False)
