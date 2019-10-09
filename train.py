import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import utils

def load_data(dataset_name):
    # load data from tfds, TODO: add support for local datasets?
    data_generators = tfds.load(name=dataset_name, batch_size=-1, data_dir="data")
    train = tf.data.Dataset.from_tensor_slices(data_generators['train']['image'])
    test = tf.data.Dataset.from_tensor_slices(data_generators['test']['image'])
    return train.concatenate(test) # we are using the whole dataset (train+test)

if __name__ == "__main__":
    args = utils.get_command_line_args()

    # loading dataset
    train_data = load_data(args.dataset)

    # preprocessing step
    if args.dataset == 'celeb_a':
        # for CelebA, they take a 140x140 centre crop of the image and resize to 32x32
        # TODO: check which interpolation they use for resizing - might be important!
        train_data = train_data.map( lambda x: tf.image.resize_with_crop_or_pad(x, crop_dim, crop_dim) 
                              ).map( lambda x: tf.image.resize(x, (scale_dim, scale_dim)) )
    train_data = train_data.map( lambda x: x / 255 ) # rescale [0,255] -> [0,1]
    
    # TRAINING LOOP CAN GO HERE
    epochs = 3
    for epoch in range(epochs):
        print("epoch", epoch)
        
        # randomly flip images
        train_data_flipped = train_data.map( lambda x: tf.image.random_flip_left_right(x) )

        # split data into batches
        train_batches = train_data_flipped.batch(128)
        num_batches = tf.data.experimental.cardinality(train_batches)
        train_batches = train_batches.shuffle(num_batches)
        
        for i, data_batch in enumerate(train_batches):
            pass
    
