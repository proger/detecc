# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

import tensorflow as tf
import numpy as np
import os
import cv2
import sys
from tqdm import tqdm
import json

from .model.test import im_detect, im_detect_fast
from .newnms.nms import soft_nms
from .nets.vgg16 import vgg16
from .nets.resnet_v1 import resnetv1

def render(class_name, dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        yield {'class': class_name, 'bbox': list(bbox.astype(int)), 'score': float(score)}

def detect(sess, net, im_file, mode='normal', cls='person', cls_ind=1):
    """Detect all objects of a single class in an image using pre-computed object proposals."""
    im = cv2.imread(im_file)

    if mode == 'fast':
        scores, boxes = im_detect_fast(sess, net, im)
    else:    
        scores, boxes = im_detect(sess, net, im)

    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    dets = soft_nms(dets, method=2)

    return dict(image=im_file, objects=list(render(cls, dets)))

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
        'res101': ('res101_faster_rcnn_iter_110000.ckpt',),
        'res152':('res152.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),
           'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),
           'coco':('coco_2014_train+coco_2014_valminusminival',)}

demonet = 'res152'
dataset = 'coco'
tfmodel = os.environ.get('TFMODEL', os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0]))

sess = tf.Session(config=tfconfig)

net = resnetv1(num_layers=152)
net.create_architecture("TEST", 81, tag='default', anchor_scales=[2, 4, 8, 16, 32])

saver = tf.train.Saver()
saver.restore(sess, tfmodel)

im_names = sys.argv[1:]

for im_name in sys.argv[1:]:
    json.dump(detect(sess, net, im_name), sys.stdout)
    sys.stdout.write('\n')
