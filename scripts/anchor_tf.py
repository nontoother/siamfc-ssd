import tensorflow as tf


class Anchor_tf():
    def __init__(self, width, height, batch_size):
        self.width = width
        self.height = height
        self.batch_size = batch_size

    def center_to_corner(self, box):
        t_1 = box[:, :, 0] - (box[:, :, 2] - 1) / 2
        t_2 = box[:, 1] - (box[:, 3] - 1) / 2
        t_3 = box[:, 0] + (box[:, 2] - 1) / 2
        t_4 = box[:, 1] + (box[:, 3] - 1) / 2
        box_temp = tf.transpose(tf.stack([t_1, t_2, t_3, t_4], axis=0), (1, 0))
        return box_temp

    def corner_to_center_gt(self, box):
        t_1 = box[:, :, 0] + (box[:, :, 2] - box[:, :, 0]) / 2
        t_2 = box[:, :, 1] + (box[:, :, 3] - box[:, :, 1]) / 2
        t_3 = (box[:, :, 2] - box[:, :, 0])
        t_4 = (box[:, :, 3] - box[:, :, 1])
        box_temp = tf.transpose(tf.stack([t_1, t_2, t_3, t_4], axis=1), (0, 2, 1))
        return box_temp

    def corner_to_center_anchors(self, box):
        t_1 = box[:, 0] + (box[:, 2] - box[:, 0]) / 2
        t_2 = box[:, 1] + (box[:, 3] - box[:, 1]) / 2
        t_3 = (box[:, 2] - box[:, 0])
        t_4 = (box[:, 3] - box[:, 1])
        box_temp = tf.transpose(tf.stack([t_1, t_2, t_3, t_4], axis=0), (1, 0))
        return box_temp

    def diff_anchor_gt(self, gt, anchors, MODE):
        # gt [x,y,w,h]
        # anchors [x,y,w,h]
        if MODE == 1:  # model 1: normal SSD with signal expanded
            t_1 = (gt[:, :, 0] - anchors[:, 0]) / anchors[:, 2] / 0.1
            t_2 = (gt[:, :, 1] - anchors[:, 1]) / anchors[:, 3] / 0.1
            t_3 = tf.log(gt[:, :, 2] / anchors[:, 2]) / 0.2
            t_4 = tf.log(gt[:, :, 3] / anchors[:, 3]) / 0.2
        elif MODE == 2:  # model 2: normal SSD without signal expanded
            t_1 = (gt[:, :, 0] - anchors[:, 0]) / anchors[:, 2]
            t_2 = (gt[:, :, 1] - anchors[:, 1]) / anchors[:, 3]
            t_3 = tf.log(gt[:, :, 2] / anchors[:, 2])
            t_4 = tf.log(gt[:, :, 3] / anchors[:, 3])
        elif MODE == 3:  # model 3: siamrpn's encode method
            t_1 = (gt[:, :, 0] - anchors[:, 0]) / (anchors[:, 2] + 0.01)
            t_2 = (gt[:, :, 1] - anchors[:, 1]) / (anchors[:, 3] + 0.01)
            t_3 = tf.log(gt[:, :, 2] / (anchors[:, 2] + 0.01))
            t_4 = tf.log(gt[:, :, 3] / (anchors[:, 3] + 0.01))
        else:
            print("please chose an encode method!")
        diff_anchors = tf.transpose(tf.stack([t_1, t_2, t_3, t_4], axis=1), (0, 2, 1))
        return diff_anchors  # [dx,dy,dw,dh]

    def decode_anchor_target(self, target_box, anchors, MODE):
        # decode, target_box: dx, dy, dw, dh
        if MODE == 1:  # model 1: normal SSD with signal expanded
            bboxes_1 = target_box[:, :, 0] * anchors[:, 2] * 0.1 + anchors[:, 0]  # anchors: cx, cy, w, h
            bboxes_2 = target_box[:, :, 1] * anchors[:, 3] * 0.1 + anchors[:, 1]
            bboxes_3 = tf.exp(target_box[:, :, 2] * 0.2) * anchors[:, 2]
            bboxes_4 = tf.exp(target_box[:, :, 3] * 0.2) * anchors[:, 3]  # [x,y,w,h], 289*4
        elif MODE == 2:  # model 2: normal SSD without signal expanded
            bboxes_1 = target_box[:, :, 0] * anchors[:, 2] + anchors[:, 0]  # anchors: cx, cy, w, h
            bboxes_2 = target_box[:, :, 1] * anchors[:, 3] + anchors[:, 1]
            bboxes_3 = tf.exp(target_box[:, :, 2]) * anchors[:, 2]
            bboxes_4 = tf.exp(target_box[:, :, 3]) * anchors[:, 3]  # [x,y,w,h], 289*4
        elif MODE == 3:  # model 3: siamrpn's encode method
            bboxes_1 = target_box[:, :, 0] * anchors[:, 2] + anchors[:, 0]  # anchors: cx, cy, w, h
            bboxes_2 = target_box[:, :, 1] * anchors[:, 3] + anchors[:, 1]
            bboxes_3 = tf.exp(target_box[:, :, 2]) * anchors[:, 2]
            bboxes_4 = tf.exp(target_box[:, :, 3]) * anchors[:, 3]  # [x,y,w,h], 289*4
        else:
            print("please chose a decode method!")
        gt_decode = tf.stack([bboxes_1, bboxes_2, bboxes_3, bboxes_4], axis=2)
        return gt_decode  # [x,y,w,h]

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

    def pos_neg_anchor(self, gt, anchors, MODE):
        # gt [x1,y1,x2,y2]
        # anchors [x1,y1,x2,y2]
        # 对anchors进行保护，保证不超出[0, 255]的范围
        zeros = tf.zeros_like(anchors)
        ones = tf.ones_like(anchors)
        all_box = tf.where(tf.less(anchors, 0), zeros, anchors)
        all_box = tf.where(tf.greater(all_box, self.width - 1), ones * (self.width - 1), all_box)

        gt = tf.reshape(gt, (self.batch_size, 1, 4))

        gt_encode = self.diff_anchor_gt(self.corner_to_center_gt(gt), self.corner_to_center_anchors(all_box), MODE)
        gt_decode = self.decode_anchor_target(gt_encode, self.corner_to_center_anchors(all_box), MODE)

        return gt_encode, all_box, gt_decode