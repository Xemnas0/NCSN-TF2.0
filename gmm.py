import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from model.modelmlp import ModelMLP
from tqdm import tqdm
from losses.losses import ssm_loss
import os, utils

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
    grid = meshgrid(x)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gmm.prob(grid), cmap='inferno', extent=[-8, 8, -8, 8], origin='lower')
    ax.title.set_text('Mixture of Gaussians')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    plt.show()
    return


def sample(gmm, nr_samples):
    s = tfd.Sample(gmm, sample_shape=nr_samples)
    return s.sample()


def visualize_samples(samples):
    plt.scatter(samples.numpy()[:, 0], samples.numpy()[:, 1], s=10)
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


def visualize_gradients(x, grads):
    U, V = grads[:, :, 1], grads[:, :, 0]
    plt.quiver(x, x, U, V)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def estimated_log_prob_grad(gmm, x_for_grads, batch_size=128, iterations=10000,):
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
def langevin_dynamics(analytic_log_prob_grad, gmm, x, sigma_i=None, alpha=0.1, T=1000):
    for t in range(T):
        z_t = tf.random.normal(shape=x.get_shape(), mean=0, stddev=1.0)
        noise = tf.sqrt(alpha) * z_t
        x = x + alpha / 2 * analytic_log_prob_grad(gmm, x, sigma_i) + noise
    return x


def annealed_langevin_dynamics(analytic_log_prob_grad, gmm, x, sigmas, eps=0.1, T=100):
    for i, sigma_i in enumerate(tqdm(sigmas)):
        alpha_i = eps * (sigma_i / sigmas[-1]) ** 2
        # idx_sigmas = tf.ones(x.get_shape()[0], dtype=tf.int32) * i
        x = langevin_dynamics(analytic_log_prob_grad, gmm, x, sigma_i, alpha_i, T) # TODO: WHERE DO WE USE SIGMA HERE IF THE GRADIENTS ARE EXACT, I.E. NOT CONDITIONAL ON SIGMA?
    return x

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # create a GMM
    gmm = gmm()

    # define grids
    x = np.linspace(-8, 8, 5, dtype=np.float32)
    x_for_grads = np.linspace(-8, 8, num=20)

    # plot density
    # visualize_density(gmm, x)

    # compute analytic gradients
    # grid = meshgrid(x_for_grads)
    # anal_grads = analytic_log_prob_grad(gmm, grid)

    # compute estimated gradients (scores)
    # estimated_log_prob_grad(gmm, x_for_grads)

    # visualize gradients
    # visualize_gradients(x_for_grads, anal_grads)

    # exact samples from the mixture
    # samples = sample(gmm, 100)

    # sampling with (annealed) Langevin dynamics
    x_init = tf.random.uniform(shape=(1280, 2), minval=-8, maxval=8)

    # Langevin dynamics
    # samples_langevin = langevin_dynamics(analytic_log_prob_grad, gmm, x_init)

    # annealed Langevin dynamics
    # TODO: THEY REPORT DIFFERENT LOW_SIGMA (SHOULD BE LOG(.))
    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(10.0),
                                           0.1,
                                           10))
    samples_annealed_langevin = annealed_langevin_dynamics(analytic_log_prob_grad, gmm, x_init, sigma_levels)

    # plot samples
    visualize_samples(samples_annealed_langevin)




# TODO: Compute grad_log_p, score, langevine_dynamics, annealed_langevine_dynamics