import tensorflow as tf
import tensorflow_datasets as tfds


def load_data(dataset_name):
    # load data from tfds, TODO: add support for local datasets?
    data_generators = tfds.load(name=dataset_name, batch_size=-1, data_dir="data", shuffle_files=True)
    train = tf.data.Dataset.from_tensor_slices(data_generators['train']['image'])
    test = tf.data.Dataset.from_tensor_slices(data_generators['test']['image'])
    return train.concatenate(test)  # TODO: split

def get_data_generator(dataset_name):
    train_data = load_data(dataset_name)

    # preprocessing step
    if dataset_name == 'celeb_a':
        # for CelebA, they take a 140x140 centre crop of the image and resize to 32x32
        # TODO: check which interpolation they use for resizing - might be important!
        train_data = train_data.map(lambda x: tf.image.resize_with_crop_or_pad(x, 140, 140)
                                    ).map(lambda x: tf.image.resize(x, (32, 32)))
    train_data = train_data.map(lambda x: x / 255)  # rescale [0,255] -> [0,1]

    if dataset_name in ["cifar10", "celeb_a"]:
        # randomly flip images
        train_data = train_data.map(lambda x: tf.image.random_flip_left_right(x))

    # split data into batches
    train_data = train_data.shuffle(1000).batch(128)

    return train_data


if __name__ == '__main__':
    pass
    # @tf.function
    # def opposite(x):
    #     if tf.random.normal([1]) > 0:
    #         return -x
    #     else:
    #         return x
    #
    #
    # data = tf.data.Dataset.from_tensor_slices(list(range(10)))
    # data = data.map(lambda x: opposite(x)).shuffle(5)
    #
    # for i in range(3):
    #     print("====")
    #     for x in data:
    #         print(x.numpy())
