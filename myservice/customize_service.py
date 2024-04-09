import os
import json
import codecs
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

from config import cfg
from frcnn.resnet_v1 import resnetv1
from frcnn.test import im_detect
from frcnn.nms.py_cpu_nms import py_cpu_nms
from collections import OrderedDict

from model_service.tfserving_model_service import TfServingBaseService

import time
import log
logger = log.getLogger(__name__)

class ObjectDetectionService(TfServingBaseService):
    def __init__(self, model_name, model_path):
        if self.is_tf_gpu_version() is True:
            print('use tf GPU version,', tf.__version__)
        else:
            print('use tf CPU version,', tf.__version__)
        
        self.model_name = model_name
        self.model_path = os.path.join(os.path.dirname(__file__), 'res101_faster_rcnn_iter_70000.ckpt.meta')
        self.input_image_key = 'images'
        self.tfmodel = os.path.join(os.path.dirname(__file__), 'res101_faster_rcnn_iter_70000.ckpt')

        self.classes_path = os.path.join(os.path.dirname(__file__), 'train_classes.txt')

        self.label_map = parse_classify_rule(os.path.join(os.path.dirname(__file__), 'classify_rule.json'))
        self.class_names = self._get_class()
        self.class_num = len(self.class_names)

        self.tfconfig = tf.ConfigProto(allow_soft_placement=True)
        self.tfconfig.gpu_options.allow_growth=True
        self.sess = tf.Session(config=self.tfconfig)

        self.net = resnetv1(num_layers=101)
        self.net.create_architecture("TEST", self.class_num,
                          tag='default', anchor_scales=[8, 16, 32])
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.tfmodel)
        print("Loaded network")
        
    def _preprocess(self, data):
        print(data)
        preprocessed_data = {}
        print("+++++++DATAITEMS========")
        print(data.items())
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                image = image.convert('RGB')
                image = np.asarray(image, dtype=np.float32)
                image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                preprocessed_data[k] = image
        return preprocessed_data
    
    def _inference(self, data):
        image = data[self.input_image_key]
        scores, boxes = im_detect(self.sess, self.net, image)
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        out_boxes = []
        out_scores = []
        out_classes = []
        for cls_ind, cls_name in enumerate(self.class_names[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = py_cpu_nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
            else:
                for i in inds:
                    out_boxes.append([round(float(v), 1) for v in dets[i, :4]])
                    out_scores.append([round(float(dets[i,-1]), 4)])
                    out_classes.append(cls_name)
        result = OrderedDict()
        if out_boxes is not None:
            detection_class_names = []
            for class_name in out_classes:
                class_name = self.label_map[class_name] + '/' + class_name
                detection_class_names.append(class_name)
            result['detection_classes'] = detection_class_names
            result['detection_scores'] = out_scores
            result['detection_boxes'] = out_boxes
        else:
            result['detection_classes'] = []
            result['detection_scores'] = []
            result['detection_boxes'] = []
        return result

    def _postprocess(self, data):
        return data
    
    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        print('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        print('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        print('postprocess time: ' + str(post_time_in_ms) + 'ms')

        print('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data
    
    def is_tf_gpu_version(self):
        from tensorflow.python.client import device_lib
        is_gpu_version = False
        devices_info = device_lib.list_local_devices()
        for device in devices_info:
            if 'GPU' == str(device.device_type):
                is_gpu_version = True
                break
        
        return is_gpu_version

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with codecs.open(classes_path, 'r', 'utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


def parse_classify_rule(json_path=''):
    with codecs.open(json_path, 'r', 'utf-8') as f:
        rule = json.load(f)
    label_map = {}
    for super_label, labels in rule.items():
        for label in labels:
            label_map[label] = super_label
    return label_map
