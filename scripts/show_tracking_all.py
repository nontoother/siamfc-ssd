#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

import os.path as osp
import sys
import os

from sacred import Experiment

ex = Experiment()

import numpy as np
from matplotlib.pyplot import imread, Rectangle

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from utils.videofig import videofig


def readbbox(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        bboxs = [[float(val) for val in line.strip().replace(' ', ',').replace('\t', ',').split(',')] for line in lines]
    return bboxs


def create_bbox(bbox, color):
    return Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                     fill=False,  # remove background\n",
                     edgecolor=color)


def set_bbox(artist, bbox):
    artist.set_xy((bbox[0], bbox[1]))
    artist.set_width(bbox[2])
    artist.set_height(bbox[3])


@ex.automain
def main():
    runname = 'SiamFC-3s-color-scratch'
    data_dir = 'new/'
    for root, dirs, files in os.walk('Logs/SiamFC/track_model_inference/SiamFC-3s-color-scratch'):
        for dir in dirs:
            videoname = dir
            track_log_dir = 'Logs/SiamFC/track_model_inference/{}/{}'.format(runname, videoname)

            track_log_dir = osp.join(track_log_dir)
            te_bboxs = readbbox(osp.join(track_log_dir, 'track_rect.txt'))      # 模型的预测框
            num_frames = len(te_bboxs)

            def redraw_fn(ind, axes):
                ind += 1
                abc = osp.join(data_dir, videoname, '{}{:05d}.jpg'.format(dir, ind + 1))
                org_img = imread(osp.join(data_dir, videoname, '{}{:05d}.jpg'.format(dir, ind + 1)))
                te_bbox = te_bboxs[ind]

                if not redraw_fn.initialized:
                    ax1 = axes

                    redraw_fn.im1 = ax1.imshow(org_img)
                    redraw_fn.bb1 = create_bbox(te_bbox, color='red')
                    ax1.add_patch(redraw_fn.bb1)

                    redraw_fn.text = ax1.text(0.03, 0.97, 'F:{}'.format(ind), fontdict={'size': 10, },
                                              ha='left', va='top',
                                              bbox={'facecolor': 'red', 'alpha': 0.7},
                                              transform=ax1.transAxes)

                    redraw_fn.initialized = True
                else:
                    redraw_fn.im1.set_array(org_img)
                    set_bbox(redraw_fn.bb1, te_bbox)
                    redraw_fn.text.set_text('F: {}'.format(ind))
                    redraw_fn.text.set_text(videoname)

            redraw_fn.initialized = False

            videofig(int(num_frames) - 1, redraw_fn)
