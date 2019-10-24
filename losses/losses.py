import tensorflow as tf


@tf.function
def dsm_loss(score, x_perturbed, x, sigmas):
    target = (x_perturbed - x) / (tf.square(sigmas))
    loss = 0.5 * tf.reduce_sum(tf.square(score+target), axis=[1,2,3], keepdims=True) * tf.square(sigmas)
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def ssm_loss(score_net, data_batch):
    sum_over = list(range(1, len(data_batch.shape)))
    v = tf.random.normal(data_batch.shape)
    with tf.GradientTape() as t:
        t.watch(data_batch)
        scores = score_net(data_batch)
        v_times_scores = tf.reduce_sum(v * scores)
    grad = t.gradient(v_times_scores, data_batch)
    loss_first = tf.reduce_sum(grad * v, axis=sum_over)
    loss_second = 0.5 * tf.reduce_sum(tf.square(scores), axis=sum_over)
    loss = tf.reduce_mean(loss_first + loss_second)

    return loss
