import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob
import cv2
import csv

from io import StringIO
from PIL import Image

import matplotlib.pyplot as plt

from utils import visualization_utils as vis_util
from utils import label_map_util

from multiprocessing.dummy import Pool as ThreadPool
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


MAX_NUMBER_OF_BOXES = 1
MINIMUM_CONFIDENCE = 0.9

PATH_TO_LABELS = 'annotations/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'test_images'
path_save = '/home/Grace/MLO-Object-Detection/result/'
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# MODEL_NAME = 'output_inference_graph'
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = '/home/Grace/Custom-Object-Detection_dataset/train0925_1000by563/output_inference_graph/frozen_inference_graph.pb'

def load_image_into_numpy_array(image_path):
    image = cv2.imread(image_path)
    
#     (im_width, im_height) = image.size
#     image = np.asarray(image)
#     print image.shape
#     exit(-1)
#     image=np.stack((image,)*3, -1)
#     print image.shape
#     return image.reshape(
#         (im_height, im_width, 3)).astype(np.uint8)
    return image

def detect_objects(image_path, path_save):
    name = os.path.basename(image_path)
    new_name = name.replace('.jpg', '_detection.jpg')
#     image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image_path)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
    print boxes
#     print boxes.shape
#     print(np.argmax(scores))
#     print scores[0], np.max(scores)
#     print len(classes[0])
#     sys.exit(-1)

    print boxes[0][0]
    ymin= boxes[0][0][0] * 1530.0
    xmin= boxes[0][0][1] * 1200.0
    ymax= boxes[0][0][2] * 1530.0
    xmax= boxes[0][0][3] * 1200.0
    print ymin, xmin, ymax,xmax
    f_list = [name, xmin, ymin,xmax,ymax]

    with open('detect.csv', mode= 'a') as location:
        w = csv.writer(location)
        w.writerow(f_list)
        
    
    var = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        min_score_thresh=MINIMUM_CONFIDENCE,
        use_normalized_coordinates=True,
        line_thickness=8)
#     fig = plt.figure()
#     fig.set_size_inches(16, 9)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)

#     plt.imshow(image_np, aspect = 'auto')
#     plt.savefig('output/{}'.format(image_path), dpi = 62)
#     plt.close(fig)
    cv2.imwrite(os.path.join(path_save, new_name), var)
    
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image-{}.jpg'.format(i)) for i in range(1, 4) ]
# TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))





TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
# Load model into memory
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('detecting...')
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        for image_path in TEST_IMAGE_PATHS:
            detect_objects(image_path, path_save)
