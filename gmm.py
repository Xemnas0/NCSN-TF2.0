import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from model.modelmlp import ModelMLP
from tqdm import tqdm
from losses.losses import ssm_loss
import os, utils

tfd = tfp.distributions


def gmm(probs, loc, scale):
    gmm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=probs),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=loc,
            scale_identity_multiplier=scale))
    return gmm


def meshgrid(x):
    y = x
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)


def visualize_density(gmm, x):
    grid = meshgrid(x)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gmm.prob(grid), cmap='inferno', extent=[-8, 8, -8, 8], origin='lower')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.savefig("density.pdf", bbox_inches="tight")
    plt.show()
    return


def sample(gmm, nr_samples):
    s = tfd.Sample(gmm, sample_shape=nr_samples)
    return s.sample()


def visualize_samples(samples, filename="samples"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples.numpy()[:, 0], samples.numpy()[:, 1], marker='.', color="black")
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.show()
    return


def analytic_log_prob_grad(gmm, x, sigma_i=None):
    x_tensor = tf.convert_to_tensor(x)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        if sigma_i is None:
            log_prob = tf.reduce_sum(gmm.log_prob(x_tensor))
        else:
            # params_dist_1 = gmm.components_distribution[0].parameters
            # params_dist_2 = gmm.components_distribution[1].parameters
            # mix_probs = gmm.mixture_distribution.probs
            normal1 = tfd.MultivariateNormalDiag(loc=[-5, -5],
                                                 scale_diag=[[sigma_i, sigma_i]])
            normal2 = tfd.MultivariateNormalDiag(loc=[5, 5],
                                                 scale_diag=[[sigma_i, sigma_i]])
            probs = list()
            probs.append(tf.math.log(tf.convert_to_tensor(0.2)) + normal1.log_prob(x_tensor))
            probs.append(tf.math.log(tf.convert_to_tensor(0.8)) + normal2.log_prob(x_tensor))

            log_prob = tf.reduce_logsumexp(tf.stack(probs, axis=0), axis=0)
    anal_gradients = t.gradient(log_prob, x_tensor)
    return anal_gradients


def visualize_gradients(x, grads, filename="gradients"):
    U, V = grads[:, :, 1], grads[:, :, 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver(x, x, U, V)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.show()


def estimated_log_prob_grad(gmm, x_for_grads, batch_size=128, iterations=10000, ):
    trained_model = train(gmm, batch_size, iterations)
    est_grads = trained_model(meshgrid(x_for_grads))
    return est_grads


def train(gmm, batch_size, total_steps):
    device = utils.get_tensorflow_device()

    # split data into batches
    model = ModelMLP(activation=tf.nn.softplus)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    step = 0
    progress_bar = tqdm(range(total_steps))
    progress_bar.set_description(f'iteration {step}/{total_steps} | current loss ?')

    with tf.device(device):  # For some reason, this makes everything faster
        avg_loss = 0
        for _ in progress_bar:
            data_batch = sample(gmm, batch_size)
            step += 1

            with tf.GradientTape(persistent=True) as t:
                current_loss = ssm_loss(model, data_batch)
                gradients = t.gradient(current_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            tf.summary.scalar('loss', float(current_loss), step=int(step))

            progress_bar.set_description(f'iteration {step}/{total_steps} | current loss {current_loss:.3f}')

            avg_loss += current_loss
            if step == total_steps:
                return model


@tf.function
def langevin_dynamics(grad_function, gmm, x, sigma_i=None, alpha=0.1, T=1000):
    for t in range(T):
        score = grad_function(gmm, x, sigma_i)
        noise = tf.sqrt(alpha) * tf.random.normal(shape=x.get_shape(), mean=0, stddev=1.0)
        x = x + (alpha / 2) * score + noise
    return x


def annealed_langevin_dynamics(grad_function, gmm, x, sigmas, eps=0.1, T=100):
    for i, sigma_i in enumerate(sigmas):
        alpha_i = eps * (sigma_i ** 2) / (sigmas[-1] ** 2)
        x = langevin_dynamics(grad_function, gmm, x, sigma_i=sigma_i, alpha=alpha_i, T=T)
    return x


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # create a GMM (probabilities, cluster centres, cluster scales)
    gmm = gmm([0.8, 0.2], [[5, 5], [-5, -5]], [1, 1])

    # define grids
    # x = np.linspace(-8, 8, 500, dtype=np.float32)
    # x_for_grads = np.linspace(-8, 8, num=20)
    #
    # # plot density
    # visualize_density(gmm, x)
    #
    # # compute analytic gradients
    # grid = meshgrid(x_for_grads)
    # anal_grads = analytic_log_prob_grad(gmm, grid)
    #
    # # compute estimated gradients (scores)
    # estimated_grads = estimated_log_prob_grad(gmm, x_for_grads)
    #
    # # visualize gradients
    # visualize_gradients(x_for_grads, anal_grads, "grad_analytic")
    # visualize_gradients(x_for_grads, estimated_grads, "grad_est")

    # exact samples from the mixture
    samples = sample(gmm, 1280)
    # visualize_samples(samples, "real_samples")
    #
    # # sampling with (annealed) Langevin dynamics
    x_init = tf.random.uniform(shape=(1280, 2), minval=-8, maxval=8)
    #
    # # Langevin dynamics
    # samples_langevin = langevin_dynamics(analytic_log_prob_grad, gmm, x_init)
    # visualize_samples(samples_langevin, "samples_langevin")
    #
    # # annealed Langevin dynamics
    # NOTE: The sigma_low is different in the original paper, but it doesn't work.
    # We take this from the original code
    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(10.0), tf.math.log(0.1), 10))
    # sigma_levels = tf.math.exp(tf.linspace(tf.math.log(5.0), tf.math.log(0.1), 10))
    # sigma_levels = np.linspace(10, 0.1, 10)
    # sigma_levels = tf.convert_to_tensor(sigma_levels, dtype=tf.float32)

    # sigma_levels = tf.clip_by_value(sigma_levels, 0.1, 10.0)
    print(sigma_levels)

    # epsilons = [10e-2, 10e-3, 10e-4, 10e-5, 10e-6, 10e-7]
    epsilons = [6 * 10e-6, 5.5 * 10e-6, 5 * 10e-6, 4.5 * 10e-6, 4 * 10e-6]
    #
    samples = []
    #
    for epsilon in epsilons:
        print(epsilon)
        samples.append(
            annealed_langevin_dynamics(analytic_log_prob_grad, gmm, x_init, sigma_levels, T=100, eps=epsilon))
    #
    # colors = ["#F7D242", "#F89111", "#D24942", "#842069", "#3B0C5C"]
    fig, ax = plt.subplots(1, len(epsilons), sharey=True, figsize=(13, 3))
    #
    for i in range(len(samples)):
        ax[i].scatter(samples[i].numpy()[:, 0], samples[i].numpy()[:, 1], s=0.5, marker='.', color='black')
    #
    ax[0].set_ylabel(r'$y$')
    ax[0].set_ylim(-10, 10)
    #
    for i, a in enumerate(ax):
        a.set_aspect('equal', 'box')
        a.set_xlabel(r'$x$')
        a.set_xlim(-10, 10)
        # a.set_ylim(-10, 10)
        a.set_title(r'$\epsilon=$' + '{0:.0e}'.format(epsilons[i]))

    # plt.savefig("samples_eps_linear_2.pdf", bbox_inches="tight")
    plt.show()

    # visualize_samples(samples_annealed_langevin, "samples_annealed_langevin_test_2")
    #
    # # # annealed Langevin dynamics
    # # sigma_levels = tf.math.exp(tf.linspace(tf.math.log(10.0), tf.math.log(0.1), 10))
    # # print(sigma_levels)
    # # samples_annealed_langevin = annealed_langevin_dynamics(analytic_log_prob_grad, gmm, x_init, sigma_levels)
    # #
    # # # plot samples
    # # visualize_samples(samples_annealed_langevin)
