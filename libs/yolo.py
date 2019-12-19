# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
import cv2

from .yolo3.model import yolo_eval
from .yolo3.utils import letterbox_image
import os
from . import config


class YOLO(object):
    def __init__(self, score=0.1, iou=0.3, gpu_num=1, **kwargs):
        self.score = score
        self.iou = iou
        self.gpu_num = gpu_num
        self.class_names = config.classes
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_anchors(self):
        anchors = [float(x) for x in config.anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = config.model_path

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = config.get_yolo_body(
                Input(shape=(None,None,config.time_step)),
                num_anchors//(2 if is_tiny_version else 3),
                num_classes
            )
            self.yolo_model.load_weights(model_path) # make sure model, anchors and classes match

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            from keras.utils import multi_gpu_model
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        assert config.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert config.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        boxed_image = letterbox_image(image, tuple(reversed(config.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })
        dets = np.concatenate([np.expand_dims(out_scores, -1), out_boxes], axis=1)
        # image_data = np.array(image, dtype='uint8')
        # image_data = cv2.cvtColor(image_data[..., config.time_step//2], cv2.COLOR_GRAY2BGR)

        # for i, c in reversed(list(enumerate(out_classes))):
        #     predicted_class = self.class_names[c]
        #     box = out_boxes[i]
        #     score = out_scores[i]

        #     top, left, bottom, right, co, si = box

        #     top = max(0, np.floor(top + 0.5).astype('int32'))
        #     left = max(0, np.floor(left + 0.5).astype('int32'))
        #     bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
        #     right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
        #     # print(label, (left, top), (right, bottom), (si, co))

        #     # if top - label_size[1] >= 0:
        #     #     text_origin = np.array([left, top - label_size[1]])
        #     # else:
        #     #     text_origin = np.array([left, top + 1])

        #     cv2.putText(image_data, f"Pred: {predicted_class}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,self.colors[c],1)
        #     cv2.rectangle(image_data, (left + i, top + i), (right - i, bottom - i), self.colors[c], 1)
        #     if config.show_dir:
        #         # Normalize direction
        #         norm = np.sqrt(co**2+si**2)
        #         co /= norm
        #         si /= norm
        #         m_x = (left+right)/2
        #         m_y = (top+bottom)/2
        #         cv2.line(image_data, (int(m_x), int(m_y)), (int(m_x+32*si), int(m_y+32*co)), self.colors[c], 1)
        #         cv2.drawMarker(image_data, (int(m_x), int(m_y)), self.colors[c], thickness=1, markerSize=8)
        return image_data, dets

    def release(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), video_fps/int(round(video_fps/10)), video_size)

    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        result, dets = yolo.detect_image(frame)
        print(dets)
    yolo.release()
    if isOutput:
        out.release()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use', default=1
    )
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default=0,
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    FLAGS = parser.parse_args()

    detect_video(YOLO(gpu_num=FLAGS.gpu_num), FLAGS.input, FLAGS.output)
