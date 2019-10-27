import os

import tensorflow as tf

import configs
import evaluation
import generate
import inpainting
import toytrain
import train
import utils

utils.manage_gpu_memory_usage()

EXPERIMENTS = {
    "train": train.main,
    "generate": generate.main,
    "inpainting": inpainting.main,
    "toytrain": toytrain.main,
    "evaluation": evaluation.main
}

if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = utils.get_command_line_args()
    configs.config_values = args

    run = EXPERIMENTS[args.experiment]

    run()
