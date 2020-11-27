import darknet
import cv2
import math


class Detector:
    """ Used to create YOLO detectors to be easily used.
    """

    def __init__(self,
                 config_file,
                 data_file,
                 weights_file,
                 thresh,
                 batch_size=1):

        self.config_file = config_file
        self.data_file = data_file
        self.weights_file = weights_file
        self.thresh = thresh
        self.batch_size = batch_size

        self.network, self.class_names, self.class_colors = darknet.load_network(self.config_file,
                                                                                 self.data_file,
                                                                                 self.weights_file,
                                                                                 batch_size=self.batch_size)

        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)

    
    def pre_processing(self, frame):
        """ Pre-process a frame to be used by the model.

        Args:
            frame (numpy.ndarray): The frame to be processed.

        Returns:
            numpy.ndarray: The processed frame.
        """

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        return frame_resized

    def detect(self, frame, print_detections=False):
        """ Detect objects in the frame using the detector.

        Args:
            frame (numpy.ndarray): Frame to be used as input to the model (detector).
            print_detections (bool, optional): Print the detections (class, conf, [x, y, w, h]). Defaults to False.

        Returns:
            tuple: The detections in this format [(class, conf, [x, y, w, h]), ...] and the processed frame.
        """

        proc_frame = self.pre_processing(frame)

        detections = darknet.detect_image(self.network,
                                          self.class_names,
                                          self.darknet_image,
                                          thresh=self.thresh)

        if print_detections and len(detections) != 0: print(detections)

        return detections, proc_frame

    
    def draw_detections(self, detections, frame, convert_to_rgb=True, classes=[]):
        """ Draw the detections detected on the frame.

        Args:
            detections (list): All the detections in this format [(class, conf, [x, y, w, h]), ...].
            frame (numpy.ndarray): The frame to draw the detections on.
            convert_to_rgb (bool, optional): Whether to convert the frame to RGB. Defaults to True.
            classes (list, optional): Limit the detections to these class names. Defaults to [].

        Returns:
            numpy.ndarray: The given frame with detections (BBs) on it.
        """

        if classes != []:
            for i, det in enumerate(detections):
                class_name = det[0]

                if class_name not in classes:
                    del detections[i]

        image = darknet.draw_boxes(detections, frame, self.class_colors)

        if convert_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    
    def get_org_frame_det(self, org_frame, proc_frame, detections, offset=(0, 0)):
        """ Convert the detection BBs coordinates from the processed frame to the original frame.

        Args:
            org_frame (numpy.ndarray): The frame the detection BBs are to be displayed/converted to.
            proc_frame (numpy.ndarray): The frame that the BBs detections are for.
            detections (list): The detections for the proc_frame, in this format [(class, conf, [x, y, w, h]), ...].
            offset (tuple, optional): (x, y) offset to the new BB coordinates. Defaults to (0, 0).

        Returns:
            list: The detections in this format [(class, conf, [x, y, w, h]), ...] to be displayed on the org_frame.
        """

        org_frame_detections = []

        for det in detections:
            name = det[0]
            conf = det[1]
            bb_x, bb_y = det[2][0], det[2][1]
            bb_w, bb_h = det[2][2], det[2][3]

            factor_y = org_frame.shape[0] / proc_frame.shape[0]
            factor_x = org_frame.shape[1] / proc_frame.shape[1]

            # The offset is so the detections can be displayed on the original frame.
            new_bb_x = (bb_x * factor_x) + offset[0]
            new_bb_y = (bb_y * factor_y) + offset[1]
            new_bb_w = bb_w * factor_x
            new_bb_h = bb_h * factor_y


            org_frame_det = (name, conf, (new_bb_x, new_bb_y, new_bb_w, new_bb_h))
            org_frame_detections.append(org_frame_det)

        return org_frame_detections
    
    def get_detection_patches(self, detections, frame, min_w, min_h, limit_classes=False, classes=[], get_highest_conf=False):
        """ Get the detected BBs patches by cropping them from the frame.

        Args:
            detections (list): All the detections in this format [(class, conf, [x, y, w, h]), ...].
            frame (numpy.ndarray): The frame the detections were detected.
            min_w (int): Minimum width of a patch/BB for it to be counted as a detection.
            min_h (int): Minimum height of a patch/BB for it to be counted as a detection.
            limit_classes (bool, optional): Limit the classes to be detected by using classes names. Defaults to False.
            classes (list, optional): List of strings, the classes you want to limit the detection to. Defaults to [].
            get_highest_conf (bool, optional): Return only the highest confidence score detection. Defaults to False.

        Returns:
            numpy.ndarray/list: The cropped patch(s).
        """

        patches = []
        
        if get_highest_conf:  # A vehicle has only 1 LP, choosing the one with the largest confidence score.
            max_conf = -1
            best_patch = None

        for det in detections:
            name = det[0]

            if limit_classes and name not in classes:
                continue

            conf = float(det[1])

            bb_x, bb_y = round(det[2][0]), round(det[2][1])
            bb_w, bb_h = round(det[2][2]), round(det[2][3])

            if bb_w < min_w or bb_h < min_h: continue

            top_l_corner_y = bb_y - (bb_h // 2)
            top_l_corner_x = bb_x - (bb_w // 2)

            if top_l_corner_y < 0: top_l_corner_y = 0
            if top_l_corner_x < 0: top_l_corner_x = 0

            patch = frame[top_l_corner_y:top_l_corner_y+bb_h, top_l_corner_x:top_l_corner_x+bb_w]

            patches.append((patch, top_l_corner_x, top_l_corner_y))

            if get_highest_conf and conf > max_conf:
                max_conf = conf
                best_patch = (patch, top_l_corner_x, top_l_corner_y)

        
        # Possibly add histogram intersection here to ensure the same vehicle is not detected multiple times/twice?
        # Or is there a much simpler way to do that.

        if get_highest_conf: return best_patch
        else: return patches

    
    def get_lp_text(self, detections, min_bottom_chars=2):
        """ Combine all LP characters detected to form the full LP text, regardless if the LP is one row or two rows.

        Args:
            detections (list): All the detections in this format [(class, conf, [x, y, w, h]), ...].
            min_bottom_chars (int, optional): Minimum num of chars needed to be below 1st char to count LP as two rows. Defaults to 2.

        Returns:
            str: The text of the full LP detected, combined detected characters in the correct order.
        """

        # Determining the most top left character of the LP (the first character).
        point1 = [0, 0]  # Top left corner coordinates.

        # Getting the euclidean distances from the top left corner to all character BBs detected.
        distances = []
        for det in detections:
            point2 = [det[2][0] - (det[2][2]/2), det[2][1] - (det[2][3]/2)]  # Top left of char BB.
            dist = math.sqrt(((point1[0] - point2[0])**2) + ((point1[1] - point2[1])**2))

            distances.append([det[0], det[1], det[2], dist])

        top_left_char = sorted(distances, key=lambda det: det[3])[0]  # Sort and take the minimum distance from the top left corner (0, 0).

        # Finding out if the LP is one row or two rows.
        # The y limit is the bottom edge of the top left character, if at least two chars are below this y_limit then the LP is two rows.
        y_limit = top_left_char[2][1] + (top_left_char[2][3] / 2)

        # Getting all chars that are below this y limit.
        below_y_limit = [det for det in detections if det[2][1] > y_limit]

        bottom_chars = len(below_y_limit)

        if bottom_chars < min_bottom_chars:  # One row LP
            lp_chars_sorted = [det[0] for det in sorted(detections, key=lambda det: det[2][0])]
            return "".join(lp_chars_sorted)

        else:  # Two rows LP
            # Sorting the characters by the y value, then finding the peak of the y values which will be the start of the second row.
            detections = sorted(detections, key=lambda det: det[2][1])


            y_peak = -1
            y_peak_i = -1

            prev_det = detections[0]  # Using first element as the initial value.

            for i in range(1, len(detections)):
                det = detections[i]

                diff = abs(det[2][1] - prev_det[2][1])
                if diff > y_peak:
                    y_peak = diff
                    y_peak_i = i
                    prev_det = det

            # Using the y peak to get the first and second row of the LP.
            upper_lp = detections[0:y_peak_i]
            lower_lp = detections[y_peak_i:]

            # Sorting using the x coordinate to make the letters in each row in correct order.
            upper_lp = [det[0] for det in sorted(upper_lp, key=lambda det: det[2][0])]
            lower_lp = [det[0] for det in sorted(lower_lp, key=lambda det: det[2][0])]

            return "".join(upper_lp + lower_lp)