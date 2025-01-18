import tensorflow as tf
import tensorflow_hub as hub

from util import *

model_name = "movenet_lightning"

print(KEYPOINT_EDGE_INDS_TO_COLOR)


if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
else:
    raise ValueError("Unsupported model name: %s" % model_name)

def movenet(input_image):
    """Runs detection on an input image.

    Args:
        input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()

    # mask = np.array([KEYPOINT_DICT[bp] for bp in ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle")])
    # filtered = keypoints_with_scores[0, 0][mask]
    # print(filtered)
    # return np.array([np.array([filtered])])

    # print(keypoints_with_scores)
    return keypoints_with_scores


image_path = "dance.gif"
image = tf.io.read_file(image_path)
image = tf.image.decode_gif(image)

num_frames, image_height, image_width, _ = image.shape
crop_region = init_crop_region(image_height, image_width)

mask = np.array([KEYPOINT_DICT[bp] for bp in ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle")])
output_images = []
filtered_points = []
# bar = display(progress(0, num_frames-1), display_id=True)
for frame_idx in range(num_frames):
  keypoints_with_scores = run_inference(
      movenet, image[frame_idx, :, :, :], crop_region,
      crop_size=[input_size, input_size])
  
  filtered = keypoints_with_scores[0, 0][mask]
  filtered_points.append(filtered)

  output_images.append(draw_prediction_on_image(
      image[frame_idx, :, :, :].numpy().astype(np.int32),
      keypoints_with_scores, crop_region=None,
      close_figure=True, output_image_height=300))
  crop_region = determine_crop_region(
      keypoints_with_scores, image_height, image_width)
#   bar.update(progress(frame_idx, num_frames-1))

# Extract essential keypoints

# print(output_images)

# Prepare gif visualization.
output = np.stack(output_images, axis=0)
print(output.shape)
to_gif(output, duration=100)
