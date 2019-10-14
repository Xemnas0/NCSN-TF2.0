import tensorflow as tf
import tensorflow_datasets as tfds
import configs

def load_data(dataset_name):
    # load data from tfds, TODO: add support for local datasets?
    data_generators = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", shuffle_files=False)
    train = tf.data.Dataset.from_tensor_slices(data_generators['train']['image'])
    test = tf.data.Dataset.from_tensor_slices(data_generators['test']['image'])
    return train, test

def preprocess(dataset_name, train, test):
    # preprocessing step
    if dataset_name == 'celeb_a':
        # for CelebA, take a 140x140 centre crop of the image and resize to 32x32
        # TODO: check which interpolation to use for resizing - might be important!
        train = train.map(lambda x: tf.image.resize_with_crop_or_pad(x, 140, 140)
                                    ).map(lambda x: tf.image.resize(x, (32, 32)))
        test = test.map(lambda x: tf.image.resize_with_crop_or_pad(x, 140, 140)
                                    ).map(lambda x: tf.image.resize(x, (32, 32)))
    train = train.map(lambda x: x / 255)  # rescale [0,255] -> [0,1]
    test = test.map(lambda x: x / 255)

    if dataset_name in ["cifar10", "celeb_a"]: # randomly flip images along the vertical axis
        train = train.map(lambda x: tf.image.random_flip_left_right(x)) 

    return train, test

def get_train_test_data(dataset_name):
    train, test = load_data(dataset_name)
    train, test = preprocess(dataset_name, train, test)

    return train, test
