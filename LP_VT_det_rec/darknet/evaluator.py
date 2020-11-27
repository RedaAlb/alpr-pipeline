import json
import os
import cv2

class Evaluator:
    """ Evaluate each stage of the ALPR pipeline, vehicle detection, LP detection, LP recognition.
    """

    def __init__(self, all_annos_file, save_v_det_wrong, save_lp_det_wrong, save_lp_rec_wrong, v_det_thresh, lp_det_thresh):

        self.all_annos_file = all_annos_file
        self.save_v_det_wrong = save_v_det_wrong
        self.save_lp_det_wrong = save_lp_det_wrong
        self.save_lp_rec_wrong = save_lp_rec_wrong
        self.v_det_thresh = v_det_thresh
        self.lp_det_thresh = lp_det_thresh

        # Loading the annotations file.
        self.load_annos(self.all_annos_file)
            
        self.v_det_tp = 0  # Number of correct (true pos) vehicle detections, vehicle(s) with LP was detected.
        self.v_det_fn = 0  # Number of incorrect (false neg) vehicle detections, where the vehicle(s) with the LP was not detected.

        self.lp_det_tp = 0
        self.lp_det_fn = 0

        self.lp_rec_tp = 0  # Number of correctly recognised LPs, where all characters were correctly recognised.
        self.lp_rec_fn = 0  # Number of incorrectly recognised LPs.

        self.v_det_wrong_samples = []  # File paths of the vehicles not detected.
        self.lp_det_wrong_samples = []
        self.lp_rec_wrong_samples = []

        self.samples_evaluated = []  # To ensure a sample is not evaluated twice.

        self.v_det_wrong_dir = "_v_det_wrong"
        self.lp_det_wrong_dir = "_lp_det_wrong"
        self.lp_rec_wrong_dir = "_lp_rec_wrong"

        # Making the directories where the wrong samples images will be saved to.
        self.make_dirs()

    def load_annos(self, annos_file):
        """ Load the annotations file.

        Args:
            annos_file (str): Path to the json annotation file.
        """

        with open(annos_file, 'r') as f:
            self.all_annos = json.load(f)

            # Combining all into 1 dict.
            temp_keys = list(self.all_annos.keys())
            for key in temp_keys:
                value = self.all_annos[key]
                self.all_annos.update(value)
                del self.all_annos[key]

    def make_dirs(self):
        """ Make directories for the wrong samples images.
        """

        if self.save_v_det_wrong:
            if not os.path.exists(self.v_det_wrong_dir): os.makedirs(self.v_det_wrong_dir)
        
        if self.save_lp_det_wrong:
            if not os.path.exists(self.lp_det_wrong_dir): os.makedirs(self.lp_det_wrong_dir)
        
        if self.save_lp_rec_wrong:
            if not os.path.exists(self.lp_rec_wrong_dir): os.makedirs(self.lp_rec_wrong_dir)
        
    
    def eval_vehicles_det(self, img_file_path, img_file_name, v_detections, frame):
        """ Evaluate the vehicles detected in the frame.

        Args:
            img_file_path (str): The relative path to the image to be evaluated.
            img_file_name (str): The image file name (with the extension).
            v_detections (list): All vehicles detected in YOLO format (class, conf, [x, y, w, h]).
            save_v_det_wrong (bool): Whether to save the wrong samples where the vehcile(s) was not detected.
            frame (numpy.ndarray): The sample frame/image, used to save when a vehicle is not detected.
        """

        if img_file_name not in self.samples_evaluated:

            image_annos = self.all_annos[img_file_name]
            gt_v_bbs =  image_annos["v_bb"]  # Ground truth vehicle BBs for each vehicle.

            for vehicle in v_detections:
                # Converting vehicle BB to top left corner.
                bb_w, bb_h = vehicle[2][2], vehicle[2][3]
                bb_x, bb_y = vehicle[2][0] - (bb_w/2), vehicle[2][1] - (bb_h/2)

                v_bb = [bb_x, bb_y, bb_w, bb_h]

                for i, gt_v_bb in enumerate(gt_v_bbs):
                    iou = self.calc_iou(v_bb, gt_v_bb)

                    if iou >= self.v_det_thresh:
                        self.v_det_tp += 1
                        del gt_v_bbs[i]


            remaining_gt_vehicles = len(gt_v_bbs)
            if remaining_gt_vehicles != 0:
                self.v_det_fn += remaining_gt_vehicles
                self.v_det_wrong_samples.append(img_file_path)
                
                if self.save_v_det_wrong: cv2.imwrite(f"{self.v_det_wrong_dir}/{img_file_name}", frame)


    def eval_lps_det(self, img_file_path, img_file_name, lp_detected, frame):
        """ Evaluate the LP detected in the frame.

        Args:
            img_file_path (str): The relative path to the image to be evaluated.
            img_file_name (str): The image file name (with the extension).
            lp_detected (tuple): The LP detected in YOLO format (class, conf, [x, y, w, h]).
            frame (numpy.ndarray): The sample frame/image, used to save when LP is not detected.
        """

        if img_file_name not in self.samples_evaluated:
            image_annos = self.all_annos[img_file_name]
            gt_lp_bbs =  image_annos["LP_bb"]  # Ground truth LP BBs for each vehicle.

            for lp in lp_detected:
                lp = lp[0]
                # Converting LP BB to top left corner.
                bb_w, bb_h = lp[2][2], lp[2][3]
                bb_x, bb_y = lp[2][0] - (bb_w/2), lp[2][1] - (bb_h/2)

                lp_bb = [bb_x, bb_y, bb_w, bb_h]

                for i, gt_lp_bb in enumerate(gt_lp_bbs):
                    iou = self.calc_iou(lp_bb, gt_lp_bb)

                    if iou >= self.lp_det_thresh:
                        self.lp_det_tp += 1
                        del gt_lp_bbs[i]

            remaining_gt_lps = len(gt_lp_bbs)
            if remaining_gt_lps != 0:
                self.lp_det_fn += remaining_gt_lps
                self.lp_det_wrong_samples.append(img_file_path)

                if self.save_lp_det_wrong and img_file_path not in self.v_det_wrong_samples:
                    cv2.imwrite(f"{self.lp_det_wrong_dir}/{img_file_name}", frame)

    
    def eval_lps_rec(self, img_file_path, img_file_name, lp_recs, frame):
        """ Evaluate the LP text recognised in the frame.

        Args:
            img_file_path (str): The relative path to the image to be evaluated.
            img_file_name (str): The image file name (with the extension).
            lp_recs (list): The full LPs text recognised.
            frame (numpy.ndarray): The sample frame/image, used to save when LP text is not detected.
        """

        if img_file_name not in self.samples_evaluated:

            image_annos = self.all_annos[img_file_name]
            gt_lps_text = image_annos["LP_chars"]

            for lp in lp_recs:
                for i, gt_lp_text in enumerate(gt_lps_text):
                    try:
                        gt_lp_text = gt_lp_text.replace("-", "")  # In case a space is represented by a "-".
                    except AttributeError: pass  # When the LP text is all numbers.


                    if lp == gt_lp_text:
                        self.lp_rec_tp += 1
                        del gt_lps_text[i]

            remaining_gt_texts = len(gt_lps_text)
            if remaining_gt_texts != 0:
                self.lp_rec_fn += remaining_gt_texts
                self.lp_rec_wrong_samples.append(img_file_path)

                if self.save_lp_rec_wrong and img_file_path not in self.lp_det_wrong_samples:
                    cv2.imwrite(f"{self.lp_rec_wrong_dir}/{img_file_name}", frame)


    def calc_iou(self, bb1, bb2):
        """ Calculate the Intersection over Union (IOU) of the two given bounding boxes.

        Args:
            bb1 (list): Bounding box 1, format: [x, y, w, h].
            bb2 (list): Bounding box 2, format: [x, y, w, h].

        Returns:
            float: The IoU.
        """

        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2
        w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)

        # No overlap
        if w_intersection <= 0 or h_intersection <= 0:
            return 0

        I = w_intersection * h_intersection
        U = w1 * h1 + w2 * h2 - I

        return I / U

    def add_done_sample(self, img_file_name, img_index, print_sample_eval):
        """ Mark a sample done so it does not get evaluated twice.

        Args:
            img_file_name (str): The image file name (with the extension).
            img_index (int): The image index.
            print_sample_eval (bool): Whether to print the evaluation for every sample.
        """

        if img_file_name not in self.samples_evaluated:
            self.samples_evaluated.append(img_file_name)
            print(f"Finished evaluating image #{img_index}, total evluated: {len(self.samples_evaluated)}")

            if print_sample_eval:
                self.print_eval_results()

    def print_eval_results(self):
        """ Print a summary of the evaluation for each stage of the pipeline.
        """

        total_v_det = self.v_det_tp + self.v_det_fn
        total_lp_det = self.lp_det_tp + self.lp_det_fn
        total_lp_rec = self.lp_rec_tp + self.lp_rec_fn

        v_det_recall = round(self.v_det_tp / total_v_det, 4) * 100
        lp_det_recall = round(self.lp_det_tp / total_lp_det, 4) * 100 
        lp_rec_recall = round(self.lp_rec_tp / total_lp_rec, 4) * 100

        if self.save_v_det_wrong: self.save_to_file(self.v_det_wrong_dir, self.v_det_wrong_samples)
        if self.save_lp_det_wrong: self.save_to_file(self.lp_det_wrong_dir, self.lp_det_wrong_samples)
        if self.save_lp_rec_wrong: self.save_to_file(self.lp_rec_wrong_dir, self.lp_rec_wrong_samples)


        print("\nEvaluation results\tTP\tFN\ttotal\trecall")
        print(f"Vehicle detection\t{self.v_det_tp}\t{self.v_det_fn}\t{total_v_det}\t{v_det_recall}")
        print(f"LP detection     \t{self.lp_det_tp}\t{self.lp_det_fn}\t{total_lp_det}\t{lp_det_recall}")
        print(f"LP recognition   \t{self.lp_rec_tp}\t{self.lp_rec_fn}\t{total_lp_rec}\t{lp_rec_recall}")

    def save_to_file(self, saving_dir, data):
        """ Save data to file .txt file.

        Args:
            saving_dir (str): The saving directory for the file.
            data (list): The data items to be saved. Each element will be on a line.
        """

        with open(f"{saving_dir}/wrong_samples.txt", "w") as file:
            data_string = "\n".join(data)
            file.write(data_string)
