from util import *
from calibrate import process_calibration, angle_between

def analyze_video (movenet, image_path, catch = "catch.jpg", legs = "legs.jpg", lean = "lean.jpg", right=True, input_size=192):
    vals = process_calibration(movenet, catch, legs, lean, right)
    
    dev = 0.25

    legs_at_catch = False
    legs_extended = False

    arms_extended = False
    arms_in = False

    lean_forward = False
    lean_backward = False

    # expected_next = None
    # current_state = None
    # 0-5 represent leg drive, hip swing, pull in arms then recovery

    image = tf.io.read_file(image_path)
    image = tf.image.decode_gif(image)

    num_frames, image_height, image_width, _ = image.shape
    crop_region = init_crop_region(image_height, image_width)

    if right: # Right side
        filt = [6, 10, 12, 16] # Shoulder, wrist, hip, ankle
    else:
        filt = [5, 9, 11, 15]


    output_images = []
    mistakes = 0
    for frame_idx in range(num_frames):
        keypoints_with_scores = run_inference(
            movenet, image[frame_idx, :, :, :], crop_region,
            crop_size=[input_size, input_size])
        
        output_images.append(draw_prediction_on_image(
            image[frame_idx, :, :, :].numpy().astype(np.int32),
            keypoints_with_scores, crop_region=None,
            close_figure=True, output_image_height=300))
        
        kps_filtered = keypoints_with_scores[0, 0, filt]

        arms_len = np.linalg.norm(kps_filtered[0, :2] - kps_filtered[1, :2])
        legs_len = np.linalg.norm(kps_filtered[2, :2] - kps_filtered[3, :2])
        angle = angle_between(kps_filtered[3, :2] - kps_filtered[2, :2], kps_filtered[0, :2] - kps_filtered[2, :2])

        if (1-dev) * vals["arms"] <= arms_len and arms_len <= (1+dev) * vals["arms"]:
            arms_extended = True
            print(f"Arms extended at frame {frame_idx}")
        elif arms_len <= 0.25 * vals["arms"]:
            arms_in = True
        
        if (1-dev) * vals["legs_catch"] <= legs_len and legs_len <= (1+dev) * vals["legs_catch"]:
            legs_at_catch = True
        elif (1-dev) * vals["legs"] <= legs_len and legs_len <= (1+dev) * vals["legs"]:
            print(f"Legs extended on frame {frame_idx}")
            legs_extended = True

        if (1-dev/2) * vals["forward"] <= angle and angle <= (1+dev/2) * vals["forward"]:
            print(f"Lean forward on frame {frame_idx}")
            lean_forward = True
        elif (1-dev/2) * vals["layback"] <= angle and angle <= (1+dev/2) * vals["layback"]:
            print(f"Lean backward on frame {frame_idx}")
            lean_backward = True

        if not ((legs_extended and lean_backward)
             or (legs_extended and not lean_backward and arms_extended)
             or (not legs_extended and lean_forward and arms_extended)):
            print(f"mistake at frame {frame_idx}")
            mistakes += 1

        arms_extended = False
        legs_extended= False
        lean_backward = False
        lean_forward = False
                   

        # crop_region = determine_crop_region(
        #     keypoints_with_scores, image_height, image_width)
        


    print(f"TOTAL MISTAKES: {mistakes}")
    
    output = np.stack(output_images, axis=0)
    to_gif(output, duration=100)

    return mistakes/num_frames

  