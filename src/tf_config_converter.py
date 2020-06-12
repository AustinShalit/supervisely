# coding: utf-8

from os.path import join

import tensorflow as tf
from google.protobuf import text_format
from supervisely_lib import logger

from object_detection.protos import pipeline_pb2 as pb


def load_sample_config(base_config_filepath):
    config = pb.TrainEvalPipelineConfig()
    with tf.gfile.GFile(base_config_filepath, 'r') as f:
        config = text_format.Merge(f.read(), config)
    return config


def default(d, key, value):
    return value if not key in d else d[key]


def remake_ssd_config(config, train_steps, n_classes, size, batch_size, checkpoint=None):
    config.model.ssd.num_classes = n_classes

    config.model.ssd.image_resizer.fixed_shape_resizer.height = size[0]
    config.model.ssd.image_resizer.fixed_shape_resizer.width = size[1]

    config.train_config.batch_size = batch_size
    config.train_config.num_steps = train_steps

    if checkpoint:
        config.train_config.fine_tune_checkpoint = checkpoint
        config.train_config.from_detection_checkpoint = True
    else:
        config.train_config.fine_tune_checkpoint = ""
        config.train_config.from_detection_checkpoint = False

    return config


def determine_tf_config(model_configuration):
    arch = model_configuration['architecture']
    backbone = model_configuration['backbone']
    logger.info('{} architecture with {} backbone will be trained.'.format(arch, backbone))
    if arch == 'ssd':
        remake_fn = remake_ssd_config
    else:
        raise NotImplemented('Unknown architecture.')
    baseconfig_fp = join('/workdir/src/base_configs', arch + '_' + backbone + '.config')
    return baseconfig_fp, remake_fn


def save_config(filepath, config):
    with open(filepath, 'w') as f:
        f.write(text_format.MessageToString(config))
