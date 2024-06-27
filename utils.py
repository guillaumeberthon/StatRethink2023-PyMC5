import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import graphviz as gr
import networkx as nx
from matplotlib import pyplot as plt
from pathlib import Path
from functools import partial
from typing import List, Callable, Optional, Tuple, Any

HERE = Path(".")


def load_data(dataset, delimiter=";"):
    fname = f"{dataset}.csv"
    data_path = HERE / "data"
    data_file = data_path / fname
    return pd.read_csv(data_file, sep=delimiter)


def crosstab(x: np.array, y: np.array, labels: list[str] = None):
    """Simple cross tabulation of two discrete vectors x and y"""
    ct = pd.crosstab(x, y)
    if labels:
        ct.index = labels
        ct.columns = labels
    return ct


def center(vals: np.ndarray) -> np.ndarray:
    return vals - np.nanmean(vals)


def standardize(vals: np.ndarray) -> np.ndarray:
    centered_vals = center(vals)
    return centered_vals / np.nanstd(centered_vals)


def convert_to_categorical(vals):
    return vals.astype("category").cat.codes.values


def logit(p: float) -> float:
    return np.log(p / (1 - p))


def invlogit(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def draw_causal_graph(
    edge_list, node_props=None, edge_props=None, graph_direction="UD"
):
    """Utility to draw a causal (directed) graph"""
    g = gr.Digraph(graph_attr={"rankdir": graph_direction})

    edge_props = {} if edge_props is None else edge_props
    for e in edge_list:
        props = edge_props[e] if e in edge_props else {}
        g.edge(e[0], e[1], **props)

    if node_props is not None:
        for name, props in node_props.items():
            g.node(name=name, **props)
    return g


def plot_scatter(xs, ys, **scatter_kwargs):
    """Draw scatter plot with consistent style (e.g. unfilled points)"""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none"}

    for k, v in defaults.items():
        val = scatter_kwargs.get(k, v)
        scatter_kwargs[k] = val

    plt.scatter(xs, ys, **scatter_kwargs)


def plot_line(xs, ys, **plot_kwargs):
    """Plot line with consistent style (e.g. bordered lines)"""
    linewidth = plot_kwargs.get("linewidth", 3)
    plot_kwargs["linewidth"] = linewidth

    # Copy settings for background
    background_plot_kwargs = {k: v for k, v in plot_kwargs.items()}
    background_plot_kwargs["linewidth"] = linewidth + 2
    background_plot_kwargs["color"] = "white"
    del background_plot_kwargs["label"]  # no legend label for background

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)


def plot_errorbar(
    xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3
):
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    """Draw thick error bars with consistent style"""
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            yerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


def plot_x_errorbar(
    xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3
):
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    """Draw thick error bars with consistent style"""
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            xerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


def plot_graph(graph, **graph_kwargs):
    """Draw a network graph.

    graph: Union[networkx.DiGraph, np.ndarray]
        if ndarray, assume `graph` is an adjacency matrix defining
        a directed graph.

    """
    # convert to networkx.DiGraph, if needed
    G = (
        nx.from_numpy_array(graph, create_using=nx.DiGraph)
        if isinstance(graph, np.ndarray)
        else graph
    )

    # Set default styling
    np.random.seed(123)  # for consistent spring-layout
    if "layout" in graph_kwargs:
        graph_kwargs["pos"] = graph_kwargs["layout"](G)

    default_graph_kwargs = {
        "node_color": "C0",
        "node_size": 500,
        "arrowsize": 30,
        "width": 3,
        "alpha": 0.7,
        "connectionstyle": "arc3,rad=0.1",
        "pos": nx.kamada_kawai_layout(G),
    }
    for k, v in default_graph_kwargs.items():
        if k not in graph_kwargs:
            graph_kwargs[k] = v

    nx.draw(G, **graph_kwargs)
    # return the node layout for consistent graphing
    return graph_kwargs["pos"]


def plot_2d_function(xrange, yrange, func, ax=None, **countour_kwargs):
    """Evaluate the function `func` over the values of xrange and yrange and
    plot the resulting value contour over that range.

    Parameters
    ----------
    xrange : np.ndarray
        The horizontal values to evaluate/plot
    yrange : p.ndarray
        The horizontal values to evaluate/plot
    func : Callable
        function of two arguments, xs and ys. Should return a single value at
        each point.
    ax : matplotlib.Axis, optional
        An optional axis to plot the function, by default None

    Returns
    -------
    contour : matplotlib.contour.QuadContourSet
    """
    resolution = len(xrange)
    xs, ys = np.meshgrid(xrange, yrange)
    xs = xs.ravel()
    ys = ys.ravel()

    value = func(xs, ys)

    if ax is not None:
        plt.sca(ax)

    return plt.contour(
        xs.reshape(resolution, resolution),
        ys.reshape(resolution, resolution),
        value.reshape(resolution, resolution),
        **countour_kwargs,
    )


def create_variables_dataframe(*variables: List[np.ndarray]) -> pd.DataFrame:
    """Converts a list of numpy arrays to a dataframe; infers column names from
    variable names
    """
    column_names = [get_variable_name(v) for v in variables]
    return pd.DataFrame(np.vstack(variables).T, columns=column_names)


def plot_pymc_distribution(distribution: pm.Distribution, 
                            draws: int = 2000, 
                            random_seed: int = 1, 
                            label: Optional[str] = None,
                            ax: Optional[plt.Axes] = None,
                              **distribution_params):
    """
    Plot a PyMC Distribution with specific distribution parameters.

    Parameters
    ----------
    distribution : pymc.Distribution
        The class of distribution to plot.
    draws : int, optional
        The number of draws/samples to generate from the distribution, by default 1000.
    random_seed : int, optional
        Random seed for reproducibility, by default 1.
    ax : matplotlib.Axes, optional
        The axes object to plot on. If None, a new axes object will be created.
    label : str, optional
        The label for the distribution, used in the legend.
    **distribution_params : dict
        Distribution-specific parameters.

    Returns
    -------
    ax : matplotlib.Axes
        The axes object associated with the plot.
    """
    # Create the distribution without a model context
    dist_instance = distribution.dist(**distribution_params)
    
    # Generate samples/draws from the distribution
    samples = pm.draw(dist_instance, draws=draws, random_seed=random_seed)
    
    # Plot the distribution
    if ax is None:
        ax = plt.gca()
    az.plot_dist(samples, ax=ax, label=label)
    
    return ax

# Example usage:
# ax = plot_pymc_distribution(pm.Gamma, alpha=2, beta=1)
# plt.show()



def savefig(filename):
    """Save a figure to the `./images` directory"""
    image_path = HERE / "images"
    if not image_path.exists():
        print(f"creating image directory: {image_path}")
        os.makedirs(image_path)

    figure_path = image_path / filename
    print(f"saving figure to {figure_path}")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")


def display_image(filename, width=600):
    """Display an image saved to the `./images` directory"""
    from IPython.display import Image, display

    return display(Image(filename=f"images/{filename}", width=width))


def simulate_2_parameter_bayesian_learning(
    x_obs,
    y_obs,
    param_a_grid,
    param_b_grid,
    true_param_a,
    true_param_b,
    model_func,
    posterior_func,
    n_posterior_samples=3,
    param_labels=None,
    data_range_x=None,
    data_range_y=None,
):
    """General function for simulating Bayesian learning in a 2-parameter model
    using grid approximation.

    Parameters
    ----------
    x_obs : np.ndarray
        The observed x values
    y_obs : np.ndarray
        The observed y values
    param_a_grid: np.ndarray
        The range of values the first model parameter in the model can take.
        Note: should have same length as param_b_grid.
    param_b_grid: np.ndarray
        The range of values the second model parameter in the model can take.
        Note: should have same length as param_a_grid.
    true_param_a: float
        The true value of the first model parameter, used for visualizing ground
        truth
    true_param_b: float
        The true value of the second model parameter, used for visualizing ground
        truth
    model_func: Callable
        A function `f` of the form `f(x, param_a, param_b)`. Evaluates the model
        given at data points x, given the current state of parameters, `param_a`
        and `param_b`. Returns a scalar output for the `y` associated with input
        `x`.
    posterior_func: Callable
        A function `f` of the form `f(x_obs, y_obs, param_grid_a, param_grid_b)
        that returns the posterior probability given the observed data and the
        range of parameters defined by `param_grid_a` and `param_grid_b`.
    n_posterior_samples: int
        The number of model functions sampled from the 2D posterior
    param_labels: Optional[list[str, str]]
        For visualization, the names of `param_a` and `param_b`, respectively
    data_range_x: Optional len-2 float sequence
        For visualization, the upper and lower bounds of the domain used for model
        evaluation
    data_range_y: Optional len-2 float sequence
        For visualization, the upper and lower bounds of the range used for model
        evaluation.
    """
    param_labels = param_labels if param_labels is not None else ["param_a", "param_b"]
    data_range_x = (x_obs.min(), x_obs.max()) if data_range_x is None else data_range_x
    data_range_y = (y_obs.min(), y_obs.max()) if data_range_y is None else data_range_y

    # NOTE: assume square parameter grid
    resolution = len(param_a_grid)

    param_a_grid, param_b_grid = np.meshgrid(param_a_grid, param_b_grid)
    param_a_grid = param_a_grid.ravel()
    param_b_grid = param_b_grid.ravel()

    posterior = posterior_func(x_obs, y_obs, param_a_grid, param_b_grid)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot Posterior over intercept and slope params
    plt.sca(axs[0])
    plt.contour(
        param_a_grid.reshape(resolution, resolution),
        param_b_grid.reshape(resolution, resolution),
        posterior.reshape(resolution, resolution),
        cmap="gray_r",
    )

    # Sample locations in parameter space according to posterior
    sample_idx = np.random.choice(
        np.arange(len(posterior)),
        p=posterior / posterior.sum(),
        size=n_posterior_samples,
    )

    param_a_list = []
    param_b_list = []
    for ii, idx in enumerate(sample_idx):
        param_a = param_a_grid[idx]
        param_b = param_b_grid[idx]
        param_a_list.append(param_a)
        param_b_list.append(param_b)

        # Add sampled parameters to posterior
        plt.scatter(param_a, param_b, s=60, c=f"C{ii}", alpha=0.75, zorder=20)

    # Add the true params to the plot for reference
    plt.scatter(
        true_param_a, true_param_b, color="k", marker="x", s=60, label="true parameters"
    )

    plt.xlabel(param_labels[0])
    plt.ylabel(param_labels[1])

    # Plot the current training data and model trends sampled from posterior
    plt.sca(axs[1])
    plt.scatter(x_obs, y_obs, s=60, c="k", alpha=0.5)

    # Plot the resulting model functions sampled from posterior
    xs = np.linspace(data_range_x[0], data_range_x[1], 100)
    for ii, (param_a, param_b) in enumerate(zip(param_a_list, param_b_list)):
        ys = model_func(xs, param_a, param_b)
        plt.plot(xs, ys, color=f"C{ii}", linewidth=4, alpha=0.5)

    groundtruth_ys = model_func(xs, true_param_a, true_param_b)
    plt.plot(
        xs, groundtruth_ys, color="k", linestyle="--", alpha=0.5, label="true trend"
    )

    plt.xlim([data_range_x[0], data_range_x[1]])
    plt.xlabel("x value")

    plt.ylim([data_range_y[0], data_range_y[1]])
    plt.ylabel("y value")

    plt.title(f"N={len(y_obs)}")
    plt.legend(loc="upper left")


# Helper function for plotting GP kernels
def plot_kernel_function(
    kernel_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    max_distance: float = 1.0,
    resolution: int = 100,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **line_kwargs: Any
) -> None:
    """
    Plots a kernel function over a range of distances.

    Parameters:
    kernel_function (Callable[[np.ndarray, np.ndarray], np.ndarray]): The kernel function to plot. It should take two 2D arrays as inputs and return a 2D covariance matrix.
    max_distance (float): The maximum distance to plot on the x-axis. Default is 1.0.
    resolution (int): The number of points to plot. Default is 100.
    label (Optional[str]): The label for the plot. Default is None.
    ax (Optional[plt.Axes]): The matplotlib Axes object to plot on. If None, uses the current Axes. Default is None.
    **line_kwargs (Any): Additional keyword arguments passed to the plot function.

    Returns:
    None

    Example:
    plot_kernel_function(quadratic_distance_kernel)
    """

    # Generate points from 0 to max_distance
    X = np.linspace(0, max_distance, resolution)[:, None]

    # Compute the covariance matrix using the kernel function
    covariance = kernel_function(X, X)

    # Generate distances for the x-axis
    distances = np.linspace(0, max_distance, resolution)

    # If an Axes object is provided, use it; otherwise, use the current Axes
    if ax is not None:
        plt.sca(ax)

    # Plot the first row of the covariance matrix against distances
    plt.plot(distances, covariance[0, :], label=label, **line_kwargs)

    # Set plot limits and labels
    plt.xlim([0, max_distance])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("|X1 - X2|")
    plt.ylabel("Covariance")

    # Add a legend if a label is provided
    if label is not None:
        plt.legend()

# GP Kernel definitions
def quadratic_distance_kernel(X0: np.ndarray, X1: np.ndarray, eta: float = 1, sigma: float = 0.5) -> np.ndarray:
    """
    Computes the quadratic distance kernel matrix between two sets of vectors.

    Parameters:
    X0 (np.ndarray): First input array of shape (n_samples_0, n_features).
    X1 (np.ndarray): Second input array of shape (n_samples_1, n_features).
    eta (float): Scaling factor for the kernel. Default is 1.
    sigma (float): Bandwidth parameter for the kernel. Default is 0.5.

    Returns:
    np.ndarray: Kernel matrix of shape (n_samples_0, n_samples_1).

    Example:
    plot_kernel_function(quadratic_distance_kernel)
    """

    # Validate inputs
    if not isinstance(X0, np.ndarray) or not isinstance(X1, np.ndarray):
        raise ValueError("X0 and X1 must be numpy arrays.")
    if X0.shape[1] != X1.shape[1]:
        raise ValueError("The number of features (columns) in X0 and X1 must be the same.")

    # Compute the L2 norm squared of each row in X0 and X1
    X0_norm_squared = np.sum(X0 ** 2, axis=1)
    X1_norm_squared = np.sum(X1 ** 2, axis=1)

    # Compute the squared Euclidean distances using the linear algebra identity:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * (x @ y)
    squared_distances = X0_norm_squared[:, np.newaxis] + X1_norm_squared[np.newaxis, :] - 2 * np.dot(X0, X1.T)

    # Compute the kernel matrix
    rho = 1 / sigma ** 2
    kernel_matrix = eta ** 2 * np.exp(-rho * squared_distances)

    return kernel_matrix


def ornstein_uhlenbeck_kernel(X0, X1, eta_squared=1, rho=4):
    distances = np.abs(X1[None, :] - X0[:, None])
    return eta_squared * np.exp(-rho * distances)


def periodic_kernel(X0, X1, eta=1, sigma=1, periodicity=.5):
    distances = np.sin((X1[None, :] - X0[:, None]) * periodicity) ** 2
    rho = 2 / sigma ** 2
    return eta ** 2 * np.exp(-rho * distances)



# Helper function for plotting Gaussian Processes
def plot_gaussian_process(
    X: np.ndarray,
    samples: Optional[np.ndarray] = None,
    mean: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    X_obs: Optional[np.ndarray] = None,
    Y_obs: Optional[np.ndarray] = None,
    uncertainty_prob: float = 0.89
) -> None:
    """
    Plots a Gaussian Process with optional samples, mean, covariance, and observed data.

    Parameters:
    X (np.ndarray): The input array of shape (n_samples,).
    samples (Optional[np.ndarray]): Samples from the Gaussian Process of shape (n_samples, n_points). Default is None.
    mean (Optional[np.ndarray]): The mean of the Gaussian Process of shape (n_points,). Default is None.
    cov (Optional[np.ndarray]): The covariance matrix of the Gaussian Process of shape (n_points, n_points). Default is None.
    X_obs (Optional[np.ndarray]): Observed input data of shape (n_obs_samples,). Default is None.
    Y_obs (Optional[np.ndarray]): Observed output data of shape (n_obs_samples,). Default is None.
    uncertainty_prob (float): The probability for the uncertainty interval. Default is 0.89.

    Returns:
    None

    """
    X = X.ravel()

    # Plot GP samples
    if samples is not None:
        for ii, sample in enumerate(samples):
            label = "GP samples" if ii == 0 else None
            plt.plot(X, sample, color=f"C{ii}", linewidth=1, label=label)

    # Add GP mean, if provided
    if mean is not None:
        mean = mean.ravel()
        plt.plot(X, mean, color='k', label='GP mean')

        # Add uncertainty around mean if covariance matrix is provided
        if cov is not None:
            z = stats.norm.ppf(1 - (1 - uncertainty_prob) / 2)
            uncertainty = z * np.sqrt(np.diag(cov))
            plt.fill_between(
                X,
                mean + uncertainty,
                mean - uncertainty,
                alpha=0.1,
                color='gray',
                zorder=1,
                label='GP uncertainty'
            )

    # Add any training data points, if provided
    if X_obs is not None and Y_obs is not None:
        plt.scatter(X_obs, Y_obs, color='k', label='observations', zorder=100, alpha=1)

    plt.xlim([X.min(), X.max()])
    plt.ylim([-5, 5])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

def plot_gaussian_process_prior(
    kernel_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_samples: int = 3,
    figsize: Tuple[int, int] = (10, 5),
    resolution: int = 100
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plots samples from a Gaussian Process prior and the kernel function.

    Parameters:
    kernel_function (Callable[[np.ndarray, np.ndarray], np.ndarray]): The kernel function to use for the Gaussian Process.
    n_samples (int): The number of samples to draw from the Gaussian Process prior. Default is 3.
    figsize (Tuple[int, int]): The size of the figure to create. Default is (10, 5).
    resolution (int): The resolution for the plot. Default is 100.

    Returns:
    Tuple[plt.Axes, plt.Axes]: The matplotlib axes objects for the Gaussian Process plot and the kernel function plot.

    Example usage:
    eta = 1
    for sigma in [0.1, .25, .5, 1, 2]:
    kernel_function = partial(quadratic_distance_kernel, eta=eta, sigma=sigma)
    axs = plot_gaussian_process_prior(kernel_function, n_samples=5)
    axs[0].set_title(f"prior: $\\eta$={eta}; $\\sigma=${sigma}")


    """
    X = np.linspace(-5, 5, resolution)[:, None]

    prior = gaussian_process_prior(X, kernel_function)
    samples = prior.rvs(n_samples)

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axs[0])
    plot_gaussian_process(X, samples=samples)
    
    plt.sca(axs[1])
    plot_kernel_function(kernel_function)
    plt.title("Kernel Function")
    return axs

def gaussian_process_prior(
    X_pred: np.ndarray,
    kernel_function: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> stats._multivariate.multivariate_normal_frozen:
    """
    Initializes a Gaussian Process prior distribution for the provided kernel function.

    Parameters:
    X_pred (np.ndarray): The input array of shape (n_samples, n_features).
    kernel_function (Callable[[np.ndarray, np.ndarray], np.ndarray]): The kernel function to use.

    Returns:
    stats._multivariate.multivariate_normal_frozen: The Gaussian Process prior distribution.
    """
    mean = np.zeros(X_pred.shape).ravel()
    cov = kernel_function(X_pred, X_pred)
    return stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)


def gaussian_process_posterior(
    X_obs: np.ndarray,
    Y_obs: np.ndarray,
    X_pred: np.ndarray,
    kernel_function,
    sigma_y: float = 1e-6,
    smoothing_factor: float = 1e-6
) -> stats._multivariate.multivariate_normal_frozen:
    """
    Computes the posterior distribution of a Gaussian Process given observations and predictions.
    
    Parameters:
    X_obs (np.ndarray): Observed input data of shape (n_obs, d).
    Y_obs (np.ndarray): Observed output data of shape (n_obs, ).
    X_pred (np.ndarray): Input data for prediction of shape (n_pred, d).
    kernel_function (callable): Kernel function that computes the covariance between points.
    sigma_y (float): Noise term for the observation covariance. Default is 1e-6.
    smoothing_factor (float): Smoothing factor added to the prediction covariance. Default is 1e-6.
    
    Returns:
    stats._multivariate.multivariate_normal_frozen: The posterior distribution of the Gaussian Process.
    """
    
    # Observation covariance matrix with noise
    K_obs = kernel_function(X_obs, X_obs) + sigma_y ** 2 * np.eye(len(X_obs))
    
    # Inverse of the observation covariance matrix
    K_obs_inv = np.linalg.inv(K_obs)

    # Covariance matrix for the prediction points with smoothing
    K_pred = kernel_function(X_pred, X_pred) + smoothing_factor * np.eye(len(X_pred))

    # Cross-covariance matrix between observations and prediction points
    K_obs_pred = kernel_function(X_obs, X_pred)

    # Compute the posterior mean
    posterior_mean = K_obs_pred.T.dot(K_obs_inv).dot(Y_obs)
    
    # Compute the posterior covariance
    posterior_cov = K_pred - K_obs_pred.T.dot(K_obs_inv).dot(K_obs_pred)
    
    # Return the posterior distribution as a multivariate normal distribution
    return stats.multivariate_normal(mean=posterior_mean.ravel(), cov=posterior_cov, allow_singular=True)

def plot_gaussian_process_posterior(
    X_obs: np.ndarray,
    Y_obs: np.ndarray,
    X_pred: np.ndarray,
    kernel_function,
    sigma_y: float = 1e-6,
    n_samples: int = 3,
    figsize: tuple = (10, 5),
    resolution: int = 100
):
    """
    Plots the posterior of a Gaussian Process given observations and prediction points.
    
    Parameters:
    X_obs (np.ndarray): Observed input data of shape (n_obs, d).
    Y_obs (np.ndarray): Observed output data of shape (n_obs, ).
    X_pred (np.ndarray): Input data for prediction of shape (n_pred, d).
    kernel_function (callable): Kernel function that computes the covariance between points.
    sigma_y (float): Noise term for the observation covariance. Default is 1e-6.
    n_samples (int): Number of samples to draw from the posterior. Default is 3.
    figsize (tuple): Size of the figure for plotting. Default is (10, 5).
    resolution (int): Resolution for the prediction grid. Default is 100.
    
    Returns:
    axs (np.ndarray): Array of matplotlib Axes objects.

    Example usage to plot Bayesian GP
    # Generate some training data
    X_pred = np.linspace(-5, 5, 100)[:, None]
    X_obs = np.linspace(-4, 4, 5)[:, None]
    Y_obs = np.sin(X_obs) ** 2 - np.cos(X_obs) #some complex function

    # Initialize the kernel function
    sigma_y = .25
    sigma_kernel = .75
    eta_kernel = 1
    kernel_function = partial(quadratic_distance_kernel, eta=eta_kernel, sigma=sigma_kernel)

    # Plot posterior
    axs = plot_gaussian_process_posterior(X_obs, Y_obs, X_pred, kernel_function, sigma_y=sigma_y, n_samples=3)
    axs[0].set_title(f"posterior");
    axs[1].set_title(f"kernel function:\n$\\eta$={eta}; $\\sigma=${sigma}")

    """
    
    # Create a prediction grid
    X = np.linspace(-5, 5, resolution)[:, None]
    
    # Compute the Gaussian Process posterior
    posterior = gaussian_process_posterior(X_obs, Y_obs, X, kernel_function, sigma_y=sigma_y)
    
    # Sample from the posterior
    samples = posterior.rvs(n_samples)

    # Set up the subplots
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Plot the Gaussian Process samples and the posterior distribution
    plt.sca(axs[0])
    plot_gaussian_process(
        X,
        samples=samples,
        mean=posterior.mean,
        cov=posterior.cov,
        X_obs=X_obs,
        Y_obs=Y_obs
    )
    
    # Plot the kernel function
    plt.sca(axs[1])
    plot_kernel_function(kernel_function, color='k')
    plt.title("Kernel Function")
    
    return axs

def plot_predictive_covariance(
    predictive: xr.Dataset,
    n_samples: int = 30,
    color: str = 'C0',
    label: Optional[str] = None
) -> None:
    """
    Plots the predictive covariance for a Gaussian Process.

    Parameters:
    predictive (dict): Dictionary containing the predictive samples. Must contain 'eta_squared' and 'rho_squared'.
    n_samples (int): Number of samples to plot. Default is 30.
    color (str): Color for the plot lines. Default is 'C0'.
    label (Optional[str]): Label for the plot lines. Default is None.

    Returns:
    None

    Example: with prior_predictive from a pymc model
    plot_predictive_covariance(prior_predictive,
                            n_samples=10,
                            label='Kernel function prior')
    plt.ylim([0, 2]);
    plt.title("Prior Covariance Functions");

    """
    
    # Extract the samples for eta_squared and rho_squared, and compute their respective values
    eta_samples = np.sqrt(predictive['eta_squared'].values[0, :n_samples])
    sigma_samples = 1 / np.sqrt(predictive['rho_squared'].values[0, :n_samples])

    # Plot the covariance function for each sample
    for idx, (eta, sigma) in enumerate(zip(eta_samples, sigma_samples)):
        # Use the label only for the first line
        current_label = label if idx == 0 else None
        
        # Define the kernel function with the current sample values
        kernel_function = partial(quadratic_distance_kernel, eta=eta, sigma=sigma)
        
        # Plot the kernel function
        plot_kernel_function(kernel_function, color=color, label=current_label, alpha=0.5, linewidth=5, max_distance=7)

#######################
# Regression test section

def main() -> None:
    """
    Main function to test the plot_pymc_distribution function.
    """
    fig, ax = plt.subplots()

    # Plot multiple distributions
    plot_pymc_distribution(pm.Gamma, alpha=2, beta=1, ax=ax, label='Gamma(alpha=2, beta=1)')
    plot_pymc_distribution(pm.Normal, mu=0, sigma=1, ax=ax, label='Normal(mu=0, sigma=1)')
    plot_pymc_distribution(pm.Exponential, lam=1, ax=ax, label='Exponential(lam=1)')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()
