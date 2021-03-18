#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.
"""Class for tracking using a track model."""
# 新cls+新reg

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os.path as osp

import numpy as np
import cv2
from cv2 import imwrite
from copy import deepcopy

from utils.infer_utils import convert_bbox_format, Rectangle
from utils.misc_utils import get_center, get
from scripts.anchor_related import Anchor

# chose model:
# model 1: normal SSD with signal expanded
# model 2: normal SSD without signal expanded
# model 3: siamrpn's encode method
MODEL = 2


class TargetState(object):
    """Represent the target state."""

    def __init__(self, bbox, search_pos, scale_idx):
        self.bbox = bbox  # (cx, cy, w, h) in the original image
        self.search_pos = search_pos  # target center position in the search image
        self.scale_idx = scale_idx  # scale index in the searched scales


class Tracker2(object):
    """Tracker based on the siamese model."""

    def __init__(self, siamese_model, model_config, track_config):
        self.siamese_model = siamese_model
        self.model_config = model_config
        self.track_config = track_config

        self.num_scales = track_config['num_scales']
        logging.info('track num scales -- {}'.format(self.num_scales))
        scales = np.arange(self.num_scales) - get_center(self.num_scales)
        self.search_factors = [self.track_config['scale_step'] ** x for x in scales]

        self.x_image_size = track_config['x_image_size']  # Search image size
        self.window = None  # Cosine window
        self.log_level = track_config['log_level']

        self.anchor_op = Anchor(17, 17)
        self.anchors = self.anchor_op.anchors
        self.anchors = self.anchor_op.corner_to_center(self.anchors)

    def track(self, sess, first_bbox, frames, logdir='/tmp'):
        """Runs tracking on a single image sequence."""
        # Get initial target bounding box and convert to center based
        bbox = convert_bbox_format(first_bbox, 'center-based')

        # Feed in the first frame image to set initial state.
        bbox_feed = [bbox.y, bbox.x, bbox.height, bbox.width]
        input_feed = [frames[0], bbox_feed]
        frame2crop_scale = self.siamese_model.initialize(sess, input_feed)

        # Storing target state
        original_target_height = bbox.height
        original_target_width = bbox.width
        search_center = np.array([get_center(self.x_image_size),
                                  get_center(self.x_image_size)])
        current_target_state = TargetState(bbox=bbox,
                                           search_pos=search_center,
                                           scale_idx=int(get_center(self.num_scales)))

        include_first = get(self.track_config, 'include_first', False)
        logging.info('Tracking include first -- {}'.format(include_first))

        # Run tracking loop
        reported_bboxs = []
        f = open('./__Data/tracking-one-Curation/Data/VID/train/a/202008280001/track.txt')
        gt_box = f.readlines()
        for i, filename in enumerate(frames):
            if i > 0 or include_first:  # We don't really want to process the first image unless intended to do so.
                # current_target_state：前一帧的bbox信息
                bbox_feed = [current_target_state.bbox.y, current_target_state.bbox.x,
                             current_target_state.bbox.height, current_target_state.bbox.width]
                input_feed = [filename, bbox_feed]

                # 将当前帧和前一帧的bbox送进模型，得到响应图，对当前帧进行滑窗检测
                outputs, metadata = self.siamese_model.inference_step(sess, input_feed)
                search_scale_list = outputs['scale_xs']  # 缩放倍数
                response = outputs['response']
                response_size = response.shape[1]
                reg_pred = outputs['reg_pred']

                # Choose the scale whole response map has the highest peak
                if self.num_scales > 1:
                    response_max = np.max(response, axis=(1, 2))
                    penalties = self.track_config['scale_penalty'] * np.ones((self.num_scales))
                    current_scale_idx = int(get_center(self.num_scales))
                    penalties[current_scale_idx] = 1.0
                    response_penalized = response_max * penalties
                    best_scale = np.argmax(response_penalized)
                else:
                    best_scale = 0

                response = response[best_scale]
                response_show = deepcopy(response)

                # decode
                bboxes = np.zeros_like(reg_pred)  # [289, 4]
                if MODEL == 1:  # model 1: normal SSD with signal expanded
                    bboxes[:, 0] = reg_pred[:, 0] * self.anchors[:, 2] * 0.1 + self.anchors[:,
                                                                               0]  # anchors: cx, cy, w, h
                    bboxes[:, 1] = reg_pred[:, 1] * self.anchors[:, 3] * 0.1 + self.anchors[:, 1]
                    bboxes[:, 2] = np.exp(reg_pred[:, 2] * 0.2) * self.anchors[:, 2]
                    bboxes[:, 3] = np.exp(reg_pred[:, 3] * 0.2) * self.anchors[:, 3]  # [x,y,w,h], 289*4
                elif MODEL == 2:  # model 2: normal SSD without signal expanded
                    bboxes[:, 0] = reg_pred[:, 0] * self.anchors[:, 2] + self.anchors[:, 0]  # anchors: cx, cy, w, h
                    bboxes[:, 1] = reg_pred[:, 1] * self.anchors[:, 3] + self.anchors[:, 1]
                    bboxes[:, 2] = np.exp(reg_pred[:, 2]) * self.anchors[:, 2]
                    bboxes[:, 3] = np.exp(reg_pred[:, 3]) * self.anchors[:, 3]  # [x,y,w,h], 289*4
                elif MODEL == 3:  # model 3: siamrpn's encode method
                    bboxes[:, 0] = reg_pred[:, 0] * self.anchors[:, 2] + self.anchors[:, 0]  # anchors: cx, cy, w, h
                    bboxes[:, 1] = reg_pred[:, 1] * self.anchors[:, 3] + self.anchors[:, 1]
                    bboxes[:, 2] = np.exp(reg_pred[:, 2]) * self.anchors[:, 2]
                    bboxes[:, 3] = np.exp(reg_pred[:, 3]) * self.anchors[:, 3]  # [x,y,w,h], 289*4
                else:
                    print("please chose a decode method!")

                with np.errstate(all='raise'):  # Raise error if something goes wrong
                    response = response - np.min(response)
                    response = response / np.sum(response)

                if self.window is None:
                    window = np.dot(np.expand_dims(np.hanning(response_size), 1),
                                    np.expand_dims(np.hanning(response_size), 0))
                    self.window = window / np.sum(window)  # normalize window
                window_influence = self.track_config['window_influence']
                response = (1 - window_influence) * response + window_influence * self.window

                # Find maximum response
                r_max, c_max = np.unravel_index(response.argmax(),
                                                response.shape)

                # Convert from crop-relative coordinates to frame coordinates
                # 坐标转换，从相对坐标转换为帧坐标
                # p_coor = np.array([r_max, c_max])  # 得分最高的点的坐标索引
                # # displacement from the center in instance final representation ...
                # disp_instance_final = p_coor - get_center(response_size)  # 最大值与上一帧目标中心的相对位移
                # # ... in instance feature space ...
                # upsample_factor = self.track_config['upsample_factor']  # upsample factor=16
                # disp_instance_feat = disp_instance_final / upsample_factor  # 映射到17*17的score map上的相对位移
                # # ... Avoid empty position ...
                # r_radius = int(response_size / upsample_factor / 2)  # r = 8
                # disp_instance_feat = np.maximum(np.minimum(disp_instance_feat, r_radius), -r_radius)  # 保证disp_instance_feat不会越界
                # # ... in instance input ...
                # disp_instance_input = disp_instance_feat * self.model_config['embed_config']['stride']
                # # ... in instance original crop (in frame coordinates)
                # disp_instance_frame = disp_instance_input / search_scale_list[best_scale]
                # # Position within frame in frame coordinates
                # y = current_target_state.bbox.y
                # x = current_target_state.bbox.x
                # temp_y = current_target_state.bbox.y
                # temp_x = current_target_state.bbox.x
                # y += disp_instance_frame[0]
                # x += disp_instance_frame[1]
                bboxes_temp = bboxes.reshape(17, 17, 4)
                box = bboxes_temp[r_max, c_max, :]
                coor_xy = np.array([box[0], box[1]])
                response_up_size = 272
                disp_instance_feat = coor_xy - get_center(response_up_size)
                upsample_factor = self.track_config['upsample_factor']
                disp_instance_feat /= upsample_factor
                r_radius = int(response_size / 2)  # r = 8
                disp_instance_input = np.maximum(np.minimum(disp_instance_feat, r_radius), -r_radius)
                # search_scale_list[best_scale] = 0.2711
                disp_instance_frame = disp_instance_input / search_scale_list[best_scale]
                # Position within frame in frame coordinates
                y = current_target_state.bbox.y
                x = current_target_state.bbox.x
                x += disp_instance_frame[0]
                y += disp_instance_frame[1]

                # Target scale damping and saturation
                # target_scale = current_target_state.bbox.height / original_target_height
                # search_factor = self.search_factors[best_scale]
                # scale_damp = self.track_config['scale_damp']  # damping factor for scale update
                # target_scale *= ((1 - scale_damp) * 1.0 + scale_damp * search_factor)
                # target_scale = np.maximum(0.2, np.minimum(5.0, target_scale))

                # Some book keeping
                # height = original_target_height * target_scale
                # width = original_target_width * target_scale
                height = box[3] / search_scale_list[best_scale]
                width = box[2] / search_scale_list[best_scale]
                current_target_state.bbox = Rectangle(x, y, width, height)
                current_target_state.scale_idx = best_scale
                current_target_state.search_pos = search_center + disp_instance_input

                assert 0 <= current_target_state.search_pos[0] < self.x_image_size, \
                    'target position in feature space should be no larger than input image size'
                assert 0 <= current_target_state.search_pos[1] < self.x_image_size, \
                    'target position in feature space should be no larger than input image size'

                if self.log_level > 0:
                    np.save(osp.join(logdir, 'num_frames.npy'), [i + 1])

                    # Select the image with the highest score scale and convert it to uint8
                    image_cropped = outputs['image_cropped'][best_scale].astype(np.uint8)
                    # Note that imwrite in cv2 assumes the image is in BGR format.
                    # However, the cropped image returned by TensorFlow is RGB.
                    # Therefore, we convert color format using cv2.cvtColor
                    imwrite(osp.join(logdir, 'image_cropped{}.jpg'.format(i)),
                            cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR))

                    np.save(osp.join(logdir, 'best_scale{}.npy'.format(i)), [best_scale])
                    np.save(osp.join(logdir, 'response{}.npy'.format(i)), response)

                    y_search, x_search = current_target_state.search_pos
                    search_scale = search_scale_list[best_scale]
                    target_height_search = height * search_scale
                    target_width_search = width * search_scale
                    bbox_search = Rectangle(x_search, y_search, target_width_search, target_height_search)
                    bbox_search = convert_bbox_format(bbox_search, 'top-left-based')
                    np.save(osp.join(logdir, 'bbox{}.npy'.format(i)),
                            [bbox_search.x, bbox_search.y, bbox_search.width, bbox_search.height])
                    ####################################################################################################
                    # 画出每一个框
                    # def iou(p1, p2, p3, p4):
                    #     s_rec1 = (p2[1] - p1[1]) * (p2[0] - p1[0])
                    #     s_rec2 = (p4[1] - p3[1]) * (p4[0] - p3[0])
                    #
                    #     sum_area = s_rec1 + s_rec2
                    #
                    #     left = max(p1[0], p3[0])
                    #     right = min(p2[0], p4[0])
                    #     top = max(p1[1], p3[1])
                    #     bottom = min(p2[1], p4[1])
                    #
                    #     if left >= right or top >= bottom:
                    #         return 0
                    #     else:
                    #         intersect = (right - left) * (bottom - top)
                    #         return (intersect / (sum_area - intersect))*1.0
                    #
                    # m = len(bboxes)
                    # gt_box_i = gt_box[i].split(',')
                    # rp1 = (int(gt_box_i[0]), int(gt_box_i[1]))
                    # rp2 = (int(gt_box_i[2]), int(gt_box_i[3]))
                    # # image = cv2.imread(filename)
                    # image_cropped = cv2.rectangle(image_cropped, rp1, rp2, (255, 255, 255))
                    # for j in range(m):
                    #     p1 = (int(np.round(bboxes[j, 0] - bboxes[j, 2] / 2)), int(np.round(bboxes[j, 1] - bboxes[j, 3] / 2)))
                    #     p2 = (int(np.round(bboxes[j, 0] + bboxes[j, 2] / 2)), int(np.round(bboxes[j, 1] + bboxes[j, 3] / 2)))
                    #     # pc = np.array([bboxes[j, 0], bboxes[j, 1]])
                    #     # disp_final = pc - get_center(response_size)
                    #     # disp_input = disp_final / search_scale_list[best_scale]
                    #     # temp_x += disp_input[0]
                    #     # temp_y += disp_input[1]  # 注意xy的顺序有没有反
                    #     # temp_w = bboxes[j, 2] / search_scale_list[best_scale]
                    #     # temp_h = bboxes[j, 3] / search_scale_list[best_scale)
                    #     # final_box = convert_bbox_format(temp_box, 'top-left-based')
                    #     riou = iou(p1, p2, rp1, rp2)
                    #     if riou >= 0.4:
                    #         image_cropped_to_write = deepcopy(image_cropped)
                    #         coor_x = int(j / 17)
                    #         coor_y = j % 17
                    #         txt_str = str(np.round(response_show[coor_x * 16, coor_y * 16], 3)) + ', (' + str(coor_x) + ',' + str(coor_y) + ')'
                    #         if reg_pred[j, 0] + reg_pred[j, 1] + reg_pred[j, 2] + reg_pred[j, 3] == 0:
                    #             image_cropped_to_write = cv2.rectangle(image_cropped_to_write, p1, p2, (0, 0, 255))
                    #             imwrite(osp.join(logdir, 'test/image_cropped{}_{}.jpg'.format(i, j)),
                    #                     cv2.cvtColor(image_cropped_to_write, cv2.COLOR_RGB2BGR))
                    #         elif riou < 0.55:
                    #             # image = cv2.rectangle(image, (int(round(final_box[0])), int(round(final_box[1]))), (int(round(final_box[2])), int(round(final_box[3]))), (0, riou * 255, 0), thickness=1)
                    #             alpha = (riou - 0.4) / 0.15
                    #             image_cropped_to_write = cv2.rectangle(image_cropped_to_write, p1, p2, (0, alpha * 255, 0))
                    #             cv2.putText(image_cropped_to_write, txt_str, (p2[0], p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
                    #             imwrite(osp.join(logdir, 'test/image_cropped{}_{}.jpg'.format(i, j)),
                    #                     cv2.cvtColor(image_cropped_to_write, cv2.COLOR_RGB2BGR))
                    #         elif riou >= 0.55:
                    #             # image = cv2.rectangle(image, (int(round(final_box[0])), int(round(final_box[1]))), (int(round(final_box[2])), int(round(final_box[3]))), (riou * 255, 0, 0), thickness=2)
                    #             alpha = (riou - 0.55) / 0.45
                    #             image_cropped_to_write = cv2.rectangle(image_cropped_to_write, p1, p2, (alpha * 255, 0, 0), thickness=2)
                    #             cv2.putText(image_cropped_to_write, txt_str, (p2[0], p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    #                         (255, 255, 255))
                    #             imwrite(osp.join(logdir, 'test/image_cropped{}_{}.jpg'.format(i, j)),
                    #                     cv2.cvtColor(image_cropped_to_write, cv2.COLOR_RGB2BGR))
                    ####################################################################################################
                    imwrite(osp.join(logdir, 'image_cropped{}.jpg'.format(i)),
                            cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR))

            reported_bbox = convert_bbox_format(current_target_state.bbox, 'top-left-based')
            image = cv2.imread(filename)
            p1 = (int(round(reported_bbox.x)), int(round(reported_bbox.y)))
            p2 = (int(round(reported_bbox.x + reported_bbox.width)), int(round(reported_bbox.y + reported_bbox.height)))
            image = cv2.rectangle(image, p1, p2, (0, 0, 255), thickness=2)
            # cv2.imshow('img', image)
            # cv2.waitKey(1)
            imwrite(osp.join(logdir, 'test/image_{}.jpg'.format(i)), image)

            reported_bboxs.append(reported_bbox)
        return reported_bboxs
