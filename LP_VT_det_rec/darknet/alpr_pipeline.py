from ctypes import *
import darknet

import random
import cv2

from detector import Detector
from evaluator import Evaluator


# TODO:
    # Introduce threading.


# Vehicle detection
v_det_config_file = "cfg/custom/2_v_det-yolov4-tiny-obj.cfg"
v_det_data_file = "data/m_data/0_v_det-obj.data"
v_det_weights = "backup_v_det/2/2_v_det-yolov4-tiny-obj_last.weights"
v_det_thresh = 0.25
vehicles_classes = ["car", "motorcycle", "bus", "truck"]  # Classes to be detected. Only needed if using full COCO trained model.
v_detector = Detector(v_det_config_file, v_det_data_file, v_det_weights, v_det_thresh)

# Minimum size of a vehicle patch in pixels.
min_v_patc_w = 40
min_v_patc_h = 40


# LP detection
lp_det_config_file = "cfg/custom/0_det-yolov4-tiny-obj.cfg"
lp_det_data_file = "data/m_data/obj.data"
lp_det_weights = "backup_lp/0/yolov4-tiny-obj_3000.weights"
lp_det_thresh = 0.65
lp_detector = Detector(lp_det_config_file, lp_det_data_file, lp_det_weights, lp_det_thresh)

# Minimum size of an lp patch.
min_lp_patch_w = 20
min_lp_patch_h = 10


# Character detection/LP recognition
char_det_config_file = "cfg/custom/5_rec-yolov4-tiny-obj.cfg"
char_det_data_file = "data/m_data/4_rec-obj.data"
char_det_weights = "backup_char/5/5_rec-yolov4-tiny-obj_last.weights"
char_det_thresh = 0.70
char_detector = Detector(char_det_config_file, char_det_data_file, char_det_weights, char_det_thresh)

# Minimum size of a character patch.
min_char_patch_w = 2
min_char_patch_h = 2


random.seed(5)  # For deterministic bbox colors.



VIDEO = False  # Whether to use video or images. If camera needed, specify the which camera source input to use.
VIDEO_PATH = "../../../../../datasets/indian/ANPR VIDEOS/28.09.2020 6/Channel_001/20200928/ch01_20200928062301.mp4"
WAIT_KEY_DELAY = 1  # Milliseconds to wait between frames (cv2.waitKey(x)).

# If VIDEO is false, then images will be taken from this img file paths .txt file. Use the "d" and "a" keys to navigate samples.
IMGS_PATH_TXT = "data/all_test_full_imgs.txt"
ALL_ANNOS_FILE = "data/all_annos.json"  # Get this json file from using the DatasetsUtils.save_all_annos_as_dict()


DISPLAY_DETECTIONS = True  # Display all detections (vehicle, LP, characters) on the frame.
DISPLAY_ORG_FRAME = False  # Display the original frame without any BBs.
DISPLAY_V_PATCH = False    # Display the cropped vehicle patches.
DISPLAY_LP_PATCH = False   # Display the cropped LP patches.
DISPLAY_CHARS_DET = False  # Display the individual character detections/BBs.


EVALUATE = True  # Whether to evaluate and record key metrics such as recognition rate and wrong samples for each pipeline stage.
PRINT_SAMPLE_EVAL = True  # Whether to print evaluation for each sample/image.
SAVE_V_DET_WRONG = False  # Whether to save the wrong samples where the vehicle detection was wrong.
SAVE_LP_DET_WRONG = False
SAVE_LP_REC_WRONG = False



if VIDEO is True:  # Using "is" because VIDEO can also be an integer of 0 or 1 when using cam as video source.
    cap = cv2.VideoCapture(VIDEO_PATH)
elif VIDEO is False:
    with open(IMGS_PATH_TXT, "r") as file:
        img_file_paths = file.read().split("\n")
        num_samples = len(img_file_paths)  # Total number of samples.
else:
    try: cap = cv2.VideoCapture(VIDEO)
    except: print("Wrong video capture source given.")


if EVALUATE:
    evaluator = Evaluator(ALL_ANNOS_FILE, SAVE_V_DET_WRONG, SAVE_LP_DET_WRONG, SAVE_LP_REC_WRONG, v_det_thresh, lp_det_thresh)



img_index = 0  # Used to navigate the images if images are used.
video_control = False  # Whether to navigate video frames or just let it play.


while True:

    # Video or camera source.
    if (VIDEO is True or type(VIDEO) is int) and not video_control:
        _, frame = cap.read()
    elif VIDEO is False:
        img_file_path = img_file_paths[img_index]
        frame = cv2.imread(img_file_path)
        if EVALUATE: img_file_name = img_file_path.split("/")[-1]



    frame_copy = frame.copy()

    # Vehicle detection
    v_detections, proc_frame = v_detector.detect(frame)  # proc -> processed.

    v_det_image = v_detector.draw_detections(v_detections, proc_frame)

    # Converting the detections to the original frame size.
    v_org_frame_detections = v_detector.get_org_frame_det(frame_copy, proc_frame, v_detections)

    if EVALUATE: evaluator.eval_vehicles_det(img_file_path, img_file_name, v_org_frame_detections, frame)
    
    # Drawing the detections on the copy of orginal frame.
    v_org_det_img = v_detector.draw_detections(v_org_frame_detections, frame_copy, convert_to_rgb=False)


    vehicle_patches = v_detector.get_detection_patches(v_org_frame_detections,
                                                       frame,
                                                       min_v_patc_w,
                                                       min_v_patc_h)

    # For LP detection and recognition evaluation.
    lps_detected = []
    lps_text_rec = []

    # For all the vehicles detected.
    for i, v_patch_data in enumerate(vehicle_patches):
        v_patch, v_bb_x, v_bb_y = v_patch_data


        # LP detection
        lp_detections, lp_proc_frame = lp_detector.detect(v_patch)
        # lp_detected_img = lp_detector.draw_detections(lp_detections, lp_proc_frame)  # Resized

        lp_org_frame_det_org = lp_detector.get_org_frame_det(v_patch, lp_proc_frame, lp_detections, offset=(v_bb_x, v_bb_y))

        if len(lp_org_frame_det_org) == 0:  # No LP detected.
            continue  # No LP detected.


        if EVALUATE: lps_detected.append(lp_org_frame_det_org)

        frame_copy = lp_detector.draw_detections(lp_org_frame_det_org, frame_copy, convert_to_rgb=False)

        if DISPLAY_V_PATCH: cv2.imshow(f"{i} Vehicle patch", v_patch)

        lp_org_frame_det = lp_detector.get_org_frame_det(v_patch, lp_proc_frame, lp_detections)

        try:
            lp_patch, lp_bb_x, lp_bb_y = lp_detector.get_detection_patches(lp_org_frame_det,
                                                                           v_patch,
                                                                           min_lp_patch_w,
                                                                           min_lp_patch_h,
                                                                           get_highest_conf=True)
        except TypeError: continue

        if DISPLAY_LP_PATCH: cv2.imshow(f"{i} LP Patch", lp_patch)

        # Character detection/LP recognition.
        char_detections, char_proc_frame = char_detector.detect(lp_patch)
        if len(char_detections) == 0: continue  # No characters detected.

        if DISPLAY_CHARS_DET:
            char_detected_img = char_detector.draw_detections(char_detections, char_proc_frame)
            cv2.imshow(f"{i} Chars detected", char_detected_img)

        lp_text = char_detector.get_lp_text(char_detections)
        cv2.putText(frame_copy, lp_text, (v_bb_x, v_bb_y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        if EVALUATE: lps_text_rec.append(lp_text)

    
    if EVALUATE:
        evaluator.eval_lps_det(img_file_path, img_file_name, lps_detected, frame)
        evaluator.eval_lps_rec(img_file_path, img_file_name, lps_text_rec, frame)
        evaluator.add_done_sample(img_file_name, img_index, PRINT_SAMPLE_EVAL)

        if not DISPLAY_DETECTIONS:
            img_index += 1  # Go through all the images.
            if img_index == num_samples: break
    

    # Displaying and keyboard input handling.
    if DISPLAY_ORG_FRAME: cv2.imshow("Original frame", frame)
    if DISPLAY_DETECTIONS: cv2.imshow("Org frame detections", frame_copy)

    key = cv2.waitKey(WAIT_KEY_DELAY)

    if key == ord("q"): break
    elif key == ord("d"):
        if video_control: _, frame = cap.read()
        elif VIDEO is False and img_index != num_samples:
            img_index += 1

        if DISPLAY_V_PATCH or DISPLAY_LP_PATCH or DISPLAY_CHARS_DET:
            cv2.destroyAllWindows()
    elif key == ord("a"):
        if VIDEO is False and num_samples != 0:
            img_index -= 1
        if DISPLAY_V_PATCH or DISPLAY_LP_PATCH or DISPLAY_CHARS_DET:
            cv2.destroyAllWindows()

    elif key == ord("p") and VIDEO is True:  # Pause video to control or unpause it to let it play.
        video_control = not video_control


if EVALUATE: evaluator.print_eval_results()

if VIDEO is True or type(VIDEO) is int:
    cap.release()

cv2.destroyAllWindows()



