#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

from datasets.sampler import Sampler
from datasets.transforms import Compose, RandomGray, RandomCrop, CenterCrop, RandomStretch
from datasets.vid import VID
from utils.misc_utils import get


class DataLoader(object):
    def __init__(self, config, is_training):
        self.config = config
        self.is_training = is_training

        preprocess_name = get(config, 'preprocessing_name', None)
        logging.info('preproces -- {}'.format(preprocess_name))

        if preprocess_name == 'siamese_fc_color':
            self.v_transform = None
            # TODO: use a single operation (tf.image.crop_and_resize) to achieve all transformations ?
            # RandomStretch：同一随机比例变大或变小，但是范围在[0.5, 1.5]之间
            # CenterCrop：裁剪或填充为目标宽高
            # RandomCrop：随机裁剪255 - 2 * 8的正方形框，随机裁出来的部分和原图大小比较像
            # Compose：从上到下将几个方法都执行一遍
            self.z_transform = Compose([RandomStretch(),
                                        CenterCrop((255 - 8, 255 - 8)),  # 模拟远近
                                        RandomCrop(255 - 2 * 8),  # 模拟上下左右移动
                                        CenterCrop((127, 127))])
            self.x_transform = Compose([RandomStretch(),
                                        CenterCrop((255 - 8, 255 - 8)),
                                        RandomCrop(255 - 2 * 8), ])
        elif preprocess_name == 'siamese_fc_gray':
            self.v_transform = RandomGray()
            self.z_transform = Compose([RandomStretch(),
                                        CenterCrop((255 - 8, 255 - 8)),
                                        RandomCrop(255 - 2 * 8),
                                        CenterCrop((127, 127))])
            self.x_transform = Compose([RandomStretch(),
                                        CenterCrop((255 - 8, 255 - 8)),
                                        RandomCrop(255 - 2 * 8), ])
        elif preprocess_name == 'None':
            self.v_transform = None
            self.z_transform = CenterCrop((127, 127))
            self.x_transform = CenterCrop((255, 255))
        else:
            raise ValueError('Preprocessing name {} was not recognized.'.format(preprocess_name))

        self.dataset_py = VID(config['input_imdb'], config['max_frame_dist'])  # dataset_py里面是全部的图片
        self.sampler = Sampler(self.dataset_py, shuffle=is_training)

    def build(self):
        self.build_dataset()
        self.build_iterator()

    def build_dataset(self):
        def sample_generator():
            for video_id in self.sampler:
                sample = self.dataset_py[video_id]
                yield sample

        def transform_fn(path1, rec1, path2, rec2):
            exemplar_file = tf.read_file(path1)
            instance_file = tf.read_file(path2)
            exemplar_image = tf.image.decode_jpeg(exemplar_file, channels=3, dct_method="INTEGER_ACCURATE")  # decode之后是个tensor，将jpeg格式图片转化成矩阵
            instance_image = tf.image.decode_jpeg(instance_file, channels=3, dct_method="INTEGER_ACCURATE")
            exemplar_rec = rec1
            instance_rec = rec2

            # 注意：exemplar image还未进行裁剪；instance rec也还未进行调整，后期需要调整。

            # if self.v_transform is not None:
            #     video = tf.stack([exemplar_image, instance_image])
            #     video = self.v_transform(video)
            #     exemplar_image = video[0]
            #     instance_image = video[2]
            #
            # if self.z_transform is not None:
            #     # 裁剪出127 * 127的exemplar image
            #     exemplar_image = self.z_transform(exemplar_image)
            #
            # if self.x_transform is not None:
            #     instance_image = self.x_transform(instance_image)
            resize_exemplar = CenterCrop((127, 127))
            exemplar_image = resize_exemplar(exemplar_image)
            resize_instance = CenterCrop((255, 255))
            instance_image = resize_instance(instance_image)

            return exemplar_image, exemplar_rec, instance_image, instance_rec

        dataset = tf.data.Dataset.from_generator(sample_generator,
                                                 output_types=(tf.string, tf.float32, tf.string, tf.float32),
                                                 output_shapes=(tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([None])))
        dataset = dataset.map(transform_fn, num_parallel_calls=self.config['prefetch_threads'])  # num_parallel_calls 并行数
        dataset = dataset.prefetch(self.config['prefetch_capacity'])  # 预先提取放在缓存中储存的数据量
        dataset = dataset.repeat()
        dataset = dataset.batch(self.config['batch_size'])

        self.dataset_tf = dataset

    def build_iterator(self):
        self.iterator = self.dataset_tf.make_one_shot_iterator()

    def get_one_batch(self):
        return self.iterator.get_next()
