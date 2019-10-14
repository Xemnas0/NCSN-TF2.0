import tensorflow as tf
import configs

@tf.function
def loss_per_batch(score, x_perturbed, x, sigmas):
    l_batch = tf.norm(sigmas * score + (x_perturbed - x) / sigmas, axis=(1, 2)) #norm over height, width
    l_batch = tf.norm(l_batch, axis=-1) # norm over channels (1 or 3)
    l_batch = tf.square(l_batch)

    #l_batch = tf.norm(sigmas * score + (x_perturbed - x) / sigmas, axis=-1)
    #l_batch += tf.norm(sigmas * score + (x_perturbed - x) / sigmas, axis=-1)
    #l_batch += tf.norm(sigmas * score + (x_perturbed - x) / sigmas, axis=-1)

    return 0.5 * tf.reduce_mean(l_batch) / configs.config_values.num_L
