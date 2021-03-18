import numpy as np
import tensorflow as tf


class Anchor():

    def __init__(self, feature_w, feature_h):
        self.w = feature_w
        self.h = feature_h
        self.base = 64  # 定义anchor的大小
        self.stride = 16
        self.scale = [3]  # 定义anchor的比例
        self.anchors = self.gen_anchors()

    def gen_single_anchor(self):
        scale = np.array(self.scale)
        s = self.base * self.base
        w = np.sqrt(s / scale)  # 约等于37
        h = w * scale  # 约等于111
        c_x = (self.stride - 1) / 2
        c_y = (self.stride - 1) / 2
        anchor = np.vstack([c_x * np.ones_like(scale), c_y * np.ones_like(scale), w, h])
        anchor = anchor.transpose()  # [x,y,w,h]
        anchor = self.center_to_corner(anchor)  # [x1,y1,x2,y2]
        anchor = anchor.astype(np.int32)
        return anchor

    def gen_anchors(self):
        anchor = self.gen_single_anchor()
        k = anchor.shape[0]
        shift_x = [x * self.stride for x in range(self.w)]  # (0, 17, 34, ..., 255)
        shift_y = [y * self.stride for y in range(self.h)]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).transpose()
        a = shifts.shape[0]
        anchor = anchor.reshape((1, k, 4))
        shifts = shifts.reshape((a, 1, 4))
        anchors = anchor + shifts
        anchors = anchors.reshape((a * k, 4))  # [x1,y1,x2,y2]
        anchors = anchors.astype(np.float32)
        return anchors

    def center_to_corner(self, box):
        box_temp = np.zeros_like(box)
        box_temp[:, 0] = box[:, 0] - (box[:, 2] - 1) / 2
        box_temp[:, 1] = box[:, 1] - (box[:, 3] - 1) / 2
        box_temp[:, 2] = box[:, 0] + (box[:, 2] - 1) / 2
        box_temp[:, 3] = box[:, 1] + (box[:, 3] - 1) / 2
        # box_temp=box_temp.astype(np.int32)
        return box_temp

    def corner_to_center(self, box):
        box_temp = np.zeros_like(box)
        box_temp[:, 0] = box[:, 0] + (box[:, 2] - box[:, 0]) / 2
        box_temp[:, 1] = box[:, 1] + (box[:, 3] - box[:, 1]) / 2
        box_temp[:, 2] = (box[:, 2] - box[:, 0])
        box_temp[:, 3] = (box[:, 3] - box[:, 1])
        # box_temp=box_temp.astype(np.int32)
        return box_temp

    def diff_anchor_gt(self, gt, anchors):
        # gt [x,y,w,h]
        # anchors [x,y,w,h]
        t_1 = (gt[0] - anchors[:, 0]) / (anchors[:, 2] + 0.01)
        t_2 = (gt[1] - anchors[:, 1]) / (anchors[:, 3] + 0.01)
        t_3 = tf.log(gt[2] / (anchors[:, 2] + 0.01))
        t_4 = tf.log(gt[3] / (anchors[:, 3] + 0.01))
        diff_anchors = tf.transpose(tf.stack([t_1, t_2, t_3, t_4], axis=0), (1, 0))
        return diff_anchors  # [dx,dy,dw,dh]

    def iou(self, box1, box2):
        """ Intersection over Union (iou)
            Args:
                box1 : [N,4]
                box2 : [K,4]
                box_type:[x1,y1,x2,y2]
            Returns:
                iou:[N,K]
        """
        N = box1.get_shape()[0]
        K = box2.get_shape()[0]
        box1 = tf.reshape(box1, (N, 1, 4)) + tf.zeros((1, K, 4))  # box1=[N,K,4]
        box2 = tf.reshape(box2, (1, K, 4)) + tf.zeros((N, 1, 4))  # box1=[N,K,4]
        x_max = tf.reduce_max(tf.stack((box1[:, :, 0], box2[:, :, 0]), axis=-1), axis=2)
        x_min = tf.reduce_min(tf.stack((box1[:, :, 2], box2[:, :, 2]), axis=-1), axis=2)
        y_max = tf.reduce_max(tf.stack((box1[:, :, 1], box2[:, :, 1]), axis=-1), axis=2)
        y_min = tf.reduce_min(tf.stack((box1[:, :, 3], box2[:, :, 3]), axis=-1), axis=2)
        tb = x_min - x_max
        lr = y_min - y_max
        zeros = tf.zeros_like(tb)
        tb = tf.where(tf.less(tb, 0), zeros, tb)
        lr = tf.where(tf.less(lr, 0), zeros, lr)
        over_square = tb * lr
        all_square = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1]) + (
                    box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1]) - over_square
        return over_square / all_square