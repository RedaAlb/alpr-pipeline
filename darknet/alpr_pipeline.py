import random
import numpy as np
import cv2

from ctypes import *
import darknet

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from detector import Detector
from evaluator import Evaluator


class ALPRPipeline:
    """ To create and run the automatic number-plate recognition (ALPR) pipeline.
    """

    def __init__(self,
                 v_det_info,
                 lp_det_info,
                 char_det_info,
                 vtc_model_path,
                 vtc_class_names,
                 video,
                 video_path,
                 imgs_path_txt,
                 all_annos_file,
                 display_detections,
                 display_org_frame,
                 display_v_patch,
                 display_lp_patch,
                 display_chars_det,
                 evaluate,
                 print_sample_eval,
                 save_v_det_wrong,
                 save_lp_det_wrong,
                 save_lp_rec_wrong,
                 same_lp_ratio=False,
                 wait_key_delay=1):
        """ Class constructor

        Args:
            v_det_info (dict): Vehicle detection. A dict with the required files and values for the darknet detector in this format:
                               (.cfg, .data, .weights, detection threshold, minimum patch width in pixels, min patch height).
            lp_det_info (dict): Same as v_det_info, but for license plate detection.
            char_det_info (dict): Same as v_det_info, but for lp character detection.
            vtc_model_path (str): Path to the vehicle type classification (vtc) model.
            vtc_class_names (list): VTC class names in alphabetical order.
            video (bool): Whether to use video or images. If camera needed, specify the which camera source input to use.
            video_path (str): Path to the video.
            imgs_path_txt (str): If video is false, then images will be taken from this img file paths .txt file.
                                 Each img file path in a new line. Use the "d" and "a" keys to navigate the samples.
            all_annos_file (str): Get this json file from using the DatasetsUtils.save_all_annos_as_dict() function.
            display_detections (bool): Display all detections (vehicle, LP, characters) on the frame.
            display_org_frame (bool): Display the original frame without any BBs.
            display_v_patch (bool): Display the cropped vehicle patches.
            display_lp_patch (bool): Display the cropped LP patches.
            display_chars_det (bool): Display the individual character detections/BBs.
            evaluate (bool): Whether to evaluate and record key metrics such as recall and wrong samples for each stage in the pipeline.
            print_sample_eval (bool): Print evaluation for each sample/image.
            save_v_det_wrong (bool): Save the wrong samples where the vehicle detection was wrong.
            save_lp_det_wrong (bool): Save the wrong samples where the licence plate detection was wrong.
            save_lp_rec_wrong (bool): Save the wrong samples where the licence plate recognition was wrong.
            same_lp_ratio (bool, optional): Keep all LP patches the same ratio (w/h) of 2.859 (avg for datasets). Defaults to False.
            wait_key_delay (int, optional): Milliseconds to wait between frames (cv2.waitKey(x)). Defaults to 1.
        """

        self.v_det_info = v_det_info
        self.lp_det_info = lp_det_info
        self.char_det_info = char_det_info
        self.vtc_model_path = vtc_model_path
        self.vtc_class_names = vtc_class_names
        self.video = video
        self.video_path = video_path
        self.imgs_path_txt = imgs_path_txt
        self.all_annos_file = all_annos_file
        self.display_detections = display_detections
        self.display_org_frame = display_org_frame
        self.display_v_patch = display_v_patch
        self.display_lp_patch = display_lp_patch
        self.display_chars_det = display_chars_det
        self.evaluate = evaluate
        self.print_sample_eval = print_sample_eval
        self.save_v_det_wrong = save_v_det_wrong
        self.save_lp_det_wrong = save_lp_det_wrong
        self.save_lp_rec_wrong = save_lp_rec_wrong
        self.same_lp_ratio = same_lp_ratio
        self.wait_key_delay = wait_key_delay

        self.v_detector = Detector(v_det_info["cfg"], v_det_info["data"], v_det_info["weights"], v_det_info["thresh"])
        self.lp_detector = Detector(lp_det_info["cfg"], lp_det_info["data"], lp_det_info["weights"], lp_det_info["thresh"])
        self.char_detector = Detector(char_det_info["cfg"], char_det_info["data"], char_det_info["weights"], char_det_info["thresh"])

        self.vtc_model = tf.keras.models.load_model(vtc_model_path)
        self.vtc_img_w, self.vtc_img_h = self.vtc_model.input.shape[1], self.vtc_model.input.shape[2]

        random.seed(5)  # For deterministic bbox colors.


    def run(self):
        """ Run the ALPR pipeline.
        """

        if self.video is True:  # Using "is" because self.video can also be an integer of 0 or 1 when using cam as video source.
            cap = cv2.VideoCapture(self.video_path)
        elif self.video is False:
            with open(self.imgs_path_txt, "r") as file:
                img_file_paths = file.read().split("\n")
                num_samples = len(img_file_paths)  # Total number of samples.
        else:
            try: cap = cv2.VideoCapture(self.video)
            except:
                print("Wrong video capture source given.")
                return


        if self.evaluate and self.video is not True:
            evaluator = Evaluator(self.all_annos_file,
                                  self.save_v_det_wrong,
                                  self.save_lp_det_wrong,
                                  self.save_lp_rec_wrong,
                                  self.v_det_info["thresh"],
                                  self.lp_det_info["thresh"])


        img_index = 0  # Used to navigate the images if images are used.
        video_control = False  # Whether to navigate video frames or just let it play.


        while True:
            # Video or camera source.
            if (self.video is True or type(self.video) is int) and not video_control:
                _, frame = cap.read()
            elif self.video is False:
                img_file_path = img_file_paths[img_index]
                frame = cv2.imread(img_file_path)
                if self.evaluate: img_file_name = img_file_path.split("/")[-1]


            frame_copy = frame.copy()

            # Vehicle detection
            v_detections, proc_frame = self.v_detector.detect(frame)  # proc -> processed.

            v_det_image = self.v_detector.draw_detections(v_detections, proc_frame)

            # Converting the detections to the original frame size.
            v_org_frame_detections = self.v_detector.get_org_frame_det(frame_copy, proc_frame, v_detections)

            if self.evaluate and self.video is not True:
                evaluator.eval_vehicles_det(img_file_path, img_file_name, v_org_frame_detections, frame)
            
            # Drawing the detections on the copy of orginal frame.
            v_org_det_img = self.v_detector.draw_detections(v_org_frame_detections, frame_copy, convert_to_rgb=False)


            vehicle_patches = self.v_detector.get_detection_patches(v_org_frame_detections,
                                                            frame,
                                                            self.v_det_info["min_patch_w"],
                                                            self.v_det_info["min_patch_h"])

            # For LP detection and recognition evaluation.
            lps_detected = []
            lps_text_rec = []

            # For all the vehicles detected.
            for i, v_patch_data in enumerate(vehicle_patches):
                v_patch, v_bb_x, v_bb_y = v_patch_data


                # Vehicle type classification
                vtc_patch = v_patch.copy()
                vtc_patch = cv2.resize(vtc_patch, (self.vtc_img_w, self.vtc_img_h))
                vtc_patch = np.array(vtc_patch).reshape((1, self.vtc_img_w, self.vtc_img_h, 3))
                vtc_patch = resnet_preprocess(vtc_patch)

                vtc_pred = list(self.vtc_model.predict(vtc_patch)[0])
                vtc_class_index = vtc_pred.index(max(vtc_pred))
                vehicle_type = self.vtc_class_names[vtc_class_index]
                cv2.putText(frame_copy, vehicle_type, (v_bb_x, v_bb_y+55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)


                # LP detection
                lp_detections, lp_proc_frame = self.lp_detector.detect(v_patch)
                # lp_detected_img = lp_detector.draw_detections(lp_detections, lp_proc_frame)  # Resized

                lp_org_frame_det_org = self.lp_detector.get_org_frame_det(v_patch, lp_proc_frame, lp_detections, offset=(v_bb_x, v_bb_y))

                if len(lp_org_frame_det_org) == 0:  # No LP detected.
                    continue

                if self.evaluate and self.video is not True: lps_detected.append(lp_org_frame_det_org)

                frame_copy = self.lp_detector.draw_detections(lp_org_frame_det_org, frame_copy, convert_to_rgb=False)

                if self.display_v_patch: cv2.imshow(f"{i} Vehicle patch", v_patch)

                lp_org_frame_det = self.lp_detector.get_org_frame_det(v_patch, lp_proc_frame, lp_detections)

                try:
                    lp_patch, lp_bb_x, lp_bb_y = self.lp_detector.get_detection_patches(lp_org_frame_det,
                                                                                v_patch,
                                                                                self.lp_det_info["min_patch_w"],
                                                                                self.lp_det_info["min_patch_h"],
                                                                                get_highest_conf=True,
                                                                                same_ratio= self.same_lp_ratio)
                except TypeError: continue

                if self.display_lp_patch: cv2.imshow(f"{i} LP Patch", lp_patch)

                # Character detection/LP recognition.
                char_detections, char_proc_frame = self.char_detector.detect(lp_patch)
                if len(char_detections) == 0: continue  # No characters detected.

                if self.display_chars_det:
                    char_detected_img = self.char_detector.draw_detections(char_detections, char_proc_frame)
                    cv2.imshow(f"{i} Chars detected", char_detected_img)

                lp_text = self.char_detector.get_lp_text(char_detections)
                cv2.putText(frame_copy, lp_text, (v_bb_x, v_bb_y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

                if self.evaluate and self.video is not True: lps_text_rec.append(lp_text)

            
            if self.evaluate and self.video is not True:
                evaluator.eval_lps_det(img_file_path, img_file_name, lps_detected, frame)
                evaluator.eval_lps_rec(img_file_path, img_file_name, lps_text_rec, frame)
                evaluator.add_done_sample(img_file_name, img_index, self.print_sample_eval)

                if not self.display_detections:
                    img_index += 1  # Go through all the images.
                    if img_index == num_samples: break
            

            # Displaying and keyboard input handling.
            if self.display_org_frame: cv2.imshow("Original frame", frame)
            if self.display_detections: cv2.imshow("Org frame detections", frame_copy)

            key = cv2.waitKey(self.wait_key_delay)

            if key == ord("q"): break
            elif key == ord("d"):
                if video_control: _, frame = cap.read()
                elif self.video is False and img_index != num_samples:
                    img_index += 1

                if self.display_v_patch or self.display_lp_patch or self.display_chars_det:
                    cv2.destroyAllWindows()
            elif key == ord("a"):
                if self.video is False and num_samples != 0:
                    img_index -= 1
                if self.display_v_patch or self.display_lp_patch or self.display_chars_det:
                    cv2.destroyAllWindows()

            elif key == ord("p") and self.video is True:  # Pause video to control or unpause it to let it play.
                video_control = not video_control


        if self.evaluate and self.video is not True: evaluator.print_eval_results()

        if self.video is True or type(self.video) is int:
            cap.release()

        cv2.destroyAllWindows()
