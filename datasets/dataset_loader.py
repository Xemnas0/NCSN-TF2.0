import tensorflow as tf
import tensorflow_datasets as tfds
import configs

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_data(dataset_name):
    # load data from tfds
    data_generators = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", shuffle_files=False)
    train = tf.data.Dataset.from_tensor_slices(data_generators['train']['image'])
    test = tf.data.Dataset.from_tensor_slices(data_generators['test']['image'])
    return train, test


def preprocess(dataset_name, data, train=True):
    data = data.map(lambda x: x / 255, num_parallel_calls=AUTOTUNE)  # rescale [0,255] -> [0,1]
    if train and dataset_name in ["cifar10"]:
        data = data.map(lambda x: tf.image.random_flip_left_right(x),
                        num_parallel_calls=AUTOTUNE)  # randomly flip along the vertical axis

    return data


def _preprocess_celeb_a(data):
    # Discard labels and landmarks
    data = data.map(lambda x: x['image'], num_parallel_calls=AUTOTUNE)
    # Take a 140x140 centre crop of the image
    data = data.map(lambda x: tf.image.resize_with_crop_or_pad(x, 140, 140), num_parallel_calls=AUTOTUNE)
    # Resize to 32x32
    data = data.map(lambda x: tf.image.resize(x, (32, 32)), num_parallel_calls=AUTOTUNE)
    # Rescale
    data = data.map(lambda x: x / 255, num_parallel_calls=AUTOTUNE)
    # Maybe cache in memory
    data = data.cache()
    # Randomly flip
    data = data.map(lambda x: tf.image.random_flip_left_right(x), num_parallel_calls=AUTOTUNE)
    return data


def get_celeb_a():
    batch_size = configs.config_values.batch_size
    data_generators = tfds.load(name='celeb_a', batch_size=batch_size, data_dir="data", shuffle_files=False)
    train = data_generators['train']
    test = data_generators['test']
    train = _preprocess_celeb_a(train)
    test = _preprocess_celeb_a(test)
    return train, test


def get_train_test_data(dataset_name):
    if dataset_name != 'celeb_a':
        train, test = load_data(dataset_name)
        train = preprocess(dataset_name, train, train=True)
        test = preprocess(dataset_name, test, train=False)
    else:
        train, test = get_celeb_a()
    return train, test


def get_data_inpainting(dataset_name, n):
    data_generator = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", split='test')
    data = tf.data.Dataset.from_tensor_slices(data_generator['image']).take(n)
    data = preprocess(dataset_name, data, train=False)

    return data.batch(1)


def get_data_k_nearest(dataset_name):
    data_generator = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", split='train', shuffle_files=False)
    data = tf.data.Dataset.from_tensor_slices(data_generator['image'])
    data = data.map(lambda x: tf.cast(x, dtype=tf.float32))

    return data
