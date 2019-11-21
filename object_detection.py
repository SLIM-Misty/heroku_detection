#!/usr/bin/env python
# coding: utf-8
 
# # Misty Object Detection Websocket server
# This script needs to be run from the tensorflow object_detection library folder.
# For information on setting up the object_detection library visit
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
 
# # Imports
import collections
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import base64
import re
from io import BytesIO
import asyncio
import websockets
import json
 
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
 
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
 
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
 
 
# ## Env setup
 
# ## Object detection imports
# Here are the imports from the object detection module.
 
from utils import label_map_util
from utils import ops as utils_ops 
from utils import visualization_utils as vis_util
 
 
# # Model preparation
 
# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
 
 
# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
 
 
# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
 
 
# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
 
 
# # Detection
 
# use this when loading saved images
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
 
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
 
 
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
 
      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})
 
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict
 
def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.
 
   Source: http://stackoverflow.com/a/9459208/284318
 
   Keyword Arguments:
   image -- PIL RGBA Image object
   color -- Tuple r, g, b (default 255, 255, 255)
 
   """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background
 
def get_bounding_box_depth_data(output_dict, image_depth_data, image_width, image_height):
  # matches the detection threshold used
  # in visualize_boxes_and_labels_on_image_array
  detection_score_threshold = 0.5
  # each bounding_box_depth_data value will be the corresponding
  # depth data for a given detection box
  bounding_box_depth_data = []
  # each object detection box is an array whose values are in the form
  # (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
  # each value is a range from 0 to 1
  # e.g. top-left-x = 0.5, image_width = 300px => pixel value is 150px
  detection_boxes = output_dict['detection_boxes']
  detection_scores = output_dict['detection_scores']
  for i in range(0, len(detection_boxes)):
    if detection_scores[i] <= detection_score_threshold:
      i = len(detection_boxes)
      break
    bounding_box_depth_data.append([])
    detection_box = detection_boxes[i]
    top_left_x = int(detection_box[0] * image_width)
    top_left_y = int(detection_box[1] * image_height)
    bottom_right_x = int(detection_box[2] * image_width)
    bottom_right_y = int(detection_box[3] * image_height)
    for y in range(top_left_y, bottom_right_y):
      for x in range(top_left_x, bottom_right_x):
        depth_value_index = (0 - x) + (y * image_width)
        if depth_value_index >= 0 and depth_value_index < len(image_depth_data):
          # these dicts are formatted to match the object schema used
          # to render the d3.js heatmap in Logging/src/views/takeDepthPicture.vue
          value = image_depth_data[depth_value_index]
          bounding_box_depth_data[i].append({
            x: x,
            y: y,
            value: value
          })
 
  return bounding_box_depth_data
 
 
def get_bounding_box_centroid_depth_data(output_dict, image_depth_data, image_width, image_height):
  detection_score_threshold = 0.5
  bounding_box_centroid_depth_data = []
  detection_boxes = output_dict['detection_boxes']
  detection_scores = output_dict['detection_scores']
  for i in range(0, len(detection_boxes)):
    if detection_scores[i] <= detection_score_threshold:
      i = len(detection_boxes)
      break
    centroid_depth = "NaN"
    detection_box = detection_boxes[i]
    top_left_x = int(detection_box[0] * image_width)
    top_left_y = int(detection_box[1] * image_height)
    bottom_right_x = int(detection_box[2] * image_width)
    bottom_right_y = int(detection_box[3] * image_height)
    offset_direction = 0
    offset_magnitude = -1
    while centroid_depth == "NaN":
      centroid_x = (bottom_right_x - top_left_x) / 2
      centroid_y = (bottom_right_y - top_left_y) / 2
      offset_direction += 1
      offset_magnitude += 1
      if offset_direction > 4:
        offset_direction = 1
      if offset_direction == 1:
        # top
        centroid_y += offset_magnitude
      if offset_direction == 2:
        # right
        centroid_x += offset_magnitude
      if offset_direction == 3:
        # bottom
        centroid_y -= offset_magnitude
      if offset_direction == 4:
        # left
        centroid_x -= offset_magnitude
      centroid_index = (0 - centroid_x) + (centroid_y * image_width)
      # the centroid is outside the depth data
      if centroid_index >= 0 and centroid_index < len(image_depth_data):
        centroid_depth = -1
      else:
        centroid_depth = image_depth_data[centroid_index]
    bounding_box_centroid_depth_data.append(centroid_depth)
 
  return bounding_box_centroid_depth_data
 
def get_turn_direction(detection_box, image_width):
  turn_direction = "none"
  threshold_offset = 10
  turn_threshold_min = int(image_width / 2) - threshold_offset
  turn_threshold_max = int(image_width / 2) + threshold_offset
 
  top_left_x = int(detection_box[0] * image_width)
  bottom_right_x = int(detection_box[2] * image_width)
  x_center = bottom_right_x - top_left_x
  if x_center < turn_threshold_min:
    turn_direction = "left"
  if x_center > turn_threshold_max:
    turn_direction = "right"
 
  return turn_direction
 
 
# parses a base64 image to a pil image
# then runs object detection on that image
# if it is a fisheye image derive depth data
# using the object detection boxes
# and returns the processed image as well as
# any relavent depth or bounding data
async def process_image(websocket, message):
  print("processing image . . . ")
  parsed_message = json.loads(message)
  image_base_64 = parsed_message['image']
  image_height = parsed_message['image_height']
  image_width = parsed_message['image_width']
  image_depth_data = parsed_message['image_depth_data']
  is_fisheye_image = parsed_message['is_fisheye_image']
  get_depth_data = parsed_message['get_depth_data']
 
  img = Image.open(BytesIO(base64.b64decode(image_base_64)))
  if is_fisheye_image:
    # remove the alpha channel, fisheye images are in rgba
    # and tensorflow just can't deal with that
    img = pure_pil_alpha_to_color_v2(img)
 
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(img)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  pil_img = Image.fromarray(image_np)
  buff = BytesIO()
  pil_img.save(buff, format="JPEG")
  processed_image_base_64 = base64.b64encode(buff.getvalue()).decode("utf-8")
 
  has_detections = len(output_dict['detection_boxes']) > 0
  # detection data is only relavent to fisheye images
  # using detection data with rgba images will result in poor depth detection
  # because the fisheye lens distorts the proportions of the image
  bounding_box_depth_data = None
  bounding_box_centroid_depth_data = None
  turn_direction = None
  if is_fisheye_image and has_detections:
    if get_depth_data:
      # retrieve the associated depth data for each bounding box
      # useful for debugging
      print("2")
      print(image_depth_data==None)
      bounding_box_depth_data = get_bounding_box_depth_data(output_dict, image_depth_data, image_width, image_height)
    # get the depth value of the centroid of each bounding box
    # if the exact centroid is 'NaN' search in each direction and
    # return the first non-NaN value. If the centroid is outside the
    # depth data bounds, return -1
    print("1")
    print(image_depth_data==None)
    bounding_box_centroid_depth_data = get_bounding_box_centroid_depth_data(output_dict, image_depth_data, image_width, image_height)
    # get a turn direction based off of the nearest detection box
    nearest_detection_index = 0
    nearest_detection_value = -1
    for i in range(0, len(bounding_box_centroid_depth_data)):
      if bounding_box_centroid_depth_data[i] > nearest_detection_value:
        nearest_detection_index = i
        nearest_detection_value = bounding_box_centroid_depth_data[i]
    turn_direction = get_turn_direction(output_dict['detection_boxes'][nearest_detection_index], image_width)
 
  return_data = {}
  return_data['image_depth_data'] = bounding_box_depth_data
  return_data['image_centroid_depth_data'] = bounding_box_centroid_depth_data
  return_data['processed_image'] = processed_image_base_64
  return_data['turn_direction'] = turn_direction
  print("image successfully processed")
  await websocket.send(json.dumps(return_data))
 
# Websocket logic
# https://websockets.readthedocs.io/en/stable/intro.html
 
async def consumer_handler(websocket, path):
  async for message in websocket:
    print('recvd messg')
    await process_image(websocket, message)
 
start_server = websockets.serve(consumer_handler, "0.0.0.0", 8765)
print('Websocket listening on port 8765!')
 
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
