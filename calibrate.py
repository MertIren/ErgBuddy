import numpy as np
import tensorflow as tf
from util import run_inference, init_crop_region

catch = "catch.jpg"
legs = "legs.jpg"
lean = "lean.jpg"
right = True

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def process_image (image, input_size):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image)

    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    return input_image


def process_calibration (movenet, catch="catch.jpg", legs="legs.jpg", lean="lean.jpg", right=True, verbose=False):
    if right: # Right side
        filt = [6, 10, 12, 16] # Shoulder, wrist, hip, ankle
    else:
        filt = [5, 9, 11, 15]

    # if right: # Right side
    #     filt = [6, 8, 10, 12, 14, 16] # Shoulder, wrist, elbow, hip, knee, ankle
    # else:
    #     filt = [5, 7, 9, 11, 13, 15]



    catch = process_image(catch, 192)

    # num_frames, image_height, image_width, _ = image.shape
    # crop_region = init_crop_region(image_height, image_width)


    legs = process_image(legs, 192)
    lean = process_image(lean, 192)

    catch_kps = movenet(catch)[0, 0, filt]
    legs_kps = movenet(legs)[0, 0, filt]
    lean_kps = movenet(lean)[0, 0, filt]

    # arms_len = angle_between()
    arms_len = np.linalg.norm(catch_kps[0, :2] - catch_kps[1, :2])
    if verbose: print(f"CALIBRATED ARMS LENGTH: {arms_len}")
    legs_len = np.linalg.norm(legs_kps[2, :2] - legs_kps[3, :2])
    if verbose: print(f"CALIBRATED LEG LENGTH: {legs_len}")
    legs_catch_len = np.linalg.norm(catch_kps[2, :2] - catch_kps[3, :2])

    forward_angle = angle_between(np.array([0, 1]), legs_kps[0, :2] - legs_kps[2, :2])
    if verbose: print(f"CALIBRATED forward angle: {forward_angle}")

    layback_angle = angle_between(np.array([0, 1]), lean_kps[0, :2] - lean_kps[2, :2])
    if verbose: print(f"CALIBRATED layback angle: {layback_angle}")


    return {"arms" : arms_len, "legs" : legs_len, "legs_catch" : legs_catch_len, 
            "forward" : forward_angle, "layback" : layback_angle}