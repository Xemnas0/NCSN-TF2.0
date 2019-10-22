import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
tfd = tfp.distributions


def gmm():
    gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(
        probs=[0.2, 0.8]),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=[[-5, -5],  # component 1
                 [5, 5]],  # component 2
            scale_identity_multiplier=[1, 1]))
    return gmm


def meshgrid(x):
    y = x
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)


def visualize_density(gmm, x):
    grid = meshgrid(np.linspace(-8, 8, 1000, dtype=np.float32))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gmm.prob(grid), cmap='inferno', extent=[-8, 8, -8, 8], origin='lower')
    ax.title.set_text('Mixture of Gaussians')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.show()


def sample(gmm, nr_samples):
    s = tfd.Sample(gmm, sample_shape=nr_samples)
    return s.sample()


def visualize_samples(samples):
    plt.scatter(samples.numpy()[:, 0], samples.numpy()[:, 1], s=10)
    plt.show()

# create a GMM
gmm = gmm()

# define grid size (x=y)
x = np.linspace(-8., 8., int(1e4), dtype=np.float32)

# plot density
# visualize_density(gmm, x)

# sample from the mixture
samples = sample(gmm, 100)

# plot samples
visualize_samples(samples)

# TODO: Compute log_p, grad_log_p, score, langevine_dynamics, annealed_langevine_dynamics