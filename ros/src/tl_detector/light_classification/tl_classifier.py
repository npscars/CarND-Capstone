import cv2
import numpy as np
import os, os.path
import rospy
import tensorflow as tf

from styx_msgs.msg import TrafficLight

from labels import LABELS
import os
import rospy

from PIL import Image, ImageDraw


class TLClassifier(object):
    def __init__(self):
        tf.reset_default_graph()
        config = tf.ConfigProto(
            # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4),
            # device_count={'GPU': 1}
        )
        # rospy.logerr("TLClassifier" + os.getcwd())
        model = os.path.join(os.getcwd(), 'light_classification/frozen_inference_graph.pb')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.session = tf.Session(config=config, graph=self.detection_graph)
        self.tl_id = LABELS[9]['id'] # 'name': u'traffic light'

    def detect_light(self, box, image):

        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        im_width, im_height = image_pil.size
        draw = ImageDraw.Draw(image_pil)
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        # box_color = (0, 255, 0)
        # draw.line([(left, top), (left, bottom), (right, bottom),
        #            (right, top), (left, top)], width=8, fill=box_color)
        # here we get a traffic classified traffic light
        traffic_light = image_pil.crop([int(left), int(top), int(right), int(bottom)])
        # TODO: find circle using Hough transform and detect color of pixels inside the circle

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.session.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # define a detection threshold with a relatively high confidence
        detection_threshold = 0.4
        # find all occurrences where probability is greater then the threshold
        max_probable_indices = []
        for idx, score in enumerate(scores[0]):
            if score >= detection_threshold and classes[0][idx] == self.tl_id:
                max_probable_indices.append(idx)
        max_probable_boxes = boxes[0][max_probable_indices]


        # TODO: (ii) detect the light of the traffic light
        for idx, box in enumerate(max_probable_boxes):
            self.detect_light(box, image)

        return TrafficLight.UNKNOWN
