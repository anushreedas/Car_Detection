import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from object_detection.utils import ops as utils_ops

utils_ops.tf = tf.compat.v1

PATH_TO_MODEL_DIR = "exported-models/pretrained_model_resnet152"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# detection_model = load_model(model_name)

PATH_TO_LABELS = "annotations/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
print(category_index)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

getdetections = {}
for image_path in glob.glob('images/predict/*'):

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections_old = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections_old.pop('num_detections'))

    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections_old.items()}

    car_detections_index = []
    for key, value in detections.items():
        if key == 'detection_classes':
            for i in range(len(value)):
                if value[i] == 3:
                    car_detections_index.append(i)

    car_detections = {key: value[0, :num_detections].numpy()[car_detections_index]
                      for key, value in detections_old.items()}


    car_detections['detection_classes'] = car_detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()


    num_detections = len(np.argwhere(car_detections['detection_scores'] >= .40))
    print('Number of cars detected: ',num_detections)
    head, tail = os.path.split(image_path)
    getdetections[tail] = num_detections

    # Draw bounding boxes and save image in detections folder
    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          car_detections['detection_boxes'],
          car_detections['detection_classes'],
          car_detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=100,
          min_score_thresh=.40,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
    head, tail = os.path.split(image_path)
    plt.savefig('detections/' + tail)

import csv
csv_file = "detections.csv"
with open(csv_file, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
               quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for key,value in getdetections.items():
        writer.writerow([key,value])
