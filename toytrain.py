import csv

from model.resnet import ToyResNet
import tensorflow as tf
import utils, os
from tqdm import tqdm
from datetime import datetime

# our files
from datasets.dataset_loader import get_train_test_data
from losses.losses import ssm_loss, dsm_loss
import configs


@tf.function
def train_one_step(model, data_batch, optimizer):
    with tf.GradientTape(persistent=True) as t:
        current_loss = ssm_loss(model, data_batch)
        gradients = t.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return current_loss


def train():
    device = utils.get_tensorflow_device()

    perturbed = False
    sigma = tf.convert_to_tensor(0.0001, dtype=tf.float32)

    # load dataset from tfds (or use downloaded version if exists)
    train_data, test_data = get_train_test_data(configs.config_values.dataset)
    num_examples = int(tf.data.experimental.cardinality(train_data))

    # split data into batches
    train_data = train_data.shuffle(1000).batch(configs.config_values.batch_size).repeat().prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    test_data = test_data.batch(configs.config_values.batch_size)

    # num_batches = int(tf.data.experimental.cardinality(train_data))
    # num_filters = {'mnist': 16, 'cifar10': 8,
    #                'celeb_a': 128}  # NOTE change mnist back to 64, cifar10 to 128 and celeb_a to 128

    # path for saving the model(s)
    save_dir = configs.config_values.checkpoint_dir + configs.config_values.dataset + 'toy1' + '/'
    # if not os.path.exists(save_dir):
    # os.makedirs(save_dir)

    start_time = datetime.now().strftime("%y%m%d-%H%M")
    log_dir = configs.config_values.log_dir + configs.config_values.dataset + 'toy1' +"/"
    summary_writer = tf.summary.create_file_writer(log_dir + start_time)

    # initialize model
    model = ToyResNet(activation=tf.nn.elu)
    # utils.print_model_summary(model)
    # declare optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # NOTE 10 times larger than in their paper

    # if resuming training, overwrite model parameters from checkpoint
    if configs.config_values.resume:
        latest_checkpoint = tf.train.latest_checkpoint(save_dir)
        print("loading model from checkpoint ", latest_checkpoint)
        step = tf.Variable(0)
        ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
        ckpt.restore(latest_checkpoint)
        step = int(step)
    else:
        step = 0

    # array of sigma levels
    # generate geometric sequence of values between sigma_low (0.01) and sigma_high (1.0)
    sigma_levels = tf.math.exp(tf.linspace(tf.math.log(configs.config_values.sigma_high),
                                           tf.math.log(configs.config_values.sigma_low),
                                           configs.config_values.num_L))

    # training loop
    print(f'dataset: {configs.config_values.dataset}, '
          f'number of examples: {num_examples}, '
          f'batch size: {configs.config_values.batch_size}\n'
          f'training...')

    total_steps = configs.config_values.steps
    progress_bar = tqdm(train_data, total=total_steps, initial=step + 1)
    progress_bar.set_description(f'iteration {step}/{total_steps} | current loss ?')

    loss_history = []
    with tf.device(device):  # For some reason, this makes everything faster
        avg_loss = 0
        for data_batch in progress_bar:
            step += 1

            if perturbed:
                sigmas = tf.reshape(sigma, shape=(data_batch.shape[0], 1, 1, 1))
                data_batch = data_batch + tf.random.normal(shape=data_batch.shape) * sigmas

            current_loss = train_one_step(model, data_batch, optimizer)
            loss_history.append(current_loss)
            progress_bar.set_description(f'iteration {step}/{total_steps} | current loss {current_loss:.3f}')

            avg_loss += current_loss
            if step % configs.config_values.checkpoint_freq == 0:
                print(f"\n Average loss: {avg_loss / configs.config_values.checkpoint_freq:.3f}")
                avg_loss = 0
            if step == total_steps:
                with open(save_dir + 'loss_history.csv', mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file, delimiter=';')
                    writer.writerows(loss_history)
                return

    # NOTE bad way to choose the best model - saving all checkpoints and then testing after


if __name__ == "__main__":
    tf.random.set_seed(2019)

    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = utils.get_command_line_args()
    configs.config_values = args

    utils.manage_gpu_memory_usage()
    train()
