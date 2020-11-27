import cv2
import os
import glob
import random
import json
import time
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
import requests

from collections import Counter


# To-do:
    # Add and make documentation consistent.

class DatasetsUtils:
    """ Provides utility functions to operate on the datasets.
    """

    def __init__(self):

        # All dataset names, which should also be the directory name for each dataset.
        self.datasets = ["caltech_cars", "english_lp", "open_alpr_eu", "aolp", "ufpr_alpr"]

        # The properties/labels to excract from the annotations.
        self.properties = ["vehicles", "position_vehicle", "type", "plate", "position_plate"]

        # The names of the keys which the labels will be saved in the dictionary for each image. Note, one for each property in 
        # properties is needed, so {prop_name1: new_name1, prop_name2: new_name2, ...}.
        self.prop_names = {"vehicles": "num_vehicles",
                           "position_vehicle": "v_bb",  # bb refers to bounding box.
                           "type": "v_type",
                           "plate": "LP_chars",
                           "position_plate": "LP_bb"}

        # This is the dataset index of all the datasets that are split into fixed train, val, and test sets, such as ufpr_alpr.
        self.split_datasets_i = [4]

        self.lp_label = 0  # Class label index for LPs.
        self.lp_chars_labels = ["0","1","2","3","4","5","6","7","8","9",
                                "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]


    # TODO: Possible rename to get_all_samples, and all references to "labels".
    def get_all_labels(self, dataset_i=0, print_log=False, subset="", prefix="", return_dict=False):
        """ Excract the needed labels from the annotation (anno) files.
        
        Parameters:
        dataset_i: The dataset index to get the labels to. Chosen from the class datasets variable.
        print_log: Whether to print the structure of the lines and values (default False).
        subset: If a dataset has a fixed train, val, test sets, specify which subset, "train", "val", "test" (default "").
        return_dict: Whether to return a dict where the key is the img file name, and value is sample data.
        
        Returns:
        list: All the labels for all the samples in the dataset, where each element is a dict for each sample, with key names
        according to prop_names, which also includes added keys LP_chars_bb and img_file_name.
        """

        # Please note, all paths are considered relative to this notebook.
        path_to_anno = f"{prefix}annotations/{self.datasets[dataset_i]}/{subset}"  # Path to the directory where all the annotations are stored.
        path_to_imgs = f"{prefix}{self.datasets[dataset_i]}/{subset}"  # Path to the direcotry where all the images are stored.

        anno_file_names = os.listdir(path_to_anno)
        img_file_names = os.listdir(path_to_imgs)
        
        all_labels = []  # Will hold all labels for all the samples in the dataset, where each element is a dict.

        if return_dict: all_labels = {}
        
        for i, anno_file in enumerate(anno_file_names):
            try: 
                with open(f"{path_to_anno}/{anno_file}", "r") as file:
                    data = file.read()
            except PermissionError as err:
                print("This dataset is split into fixed train, test, and val sets, please provide an additional argument of\nsubset=\"training|testing|validation\".")
                return None

            lines = data.replace("\t", "").replace("-","").split("\n")

            if print_log:
                print("\n", anno_file, len(lines))

            labels = {}  # Will hold all the labels with keys as set in prop_names.
            LP_chars_pos = []  # Will hold a 2D array of all the LP character positions for all vehicles in the image.
            
            for line in lines:
                line_split = line.split(":")

                try:
                    prop = line_split[0].strip()
                    data = line_split[1].strip()
                except IndexError: continue  # For empty lines.            

                if prop in self.properties:
                    # Cleaning up the data
                    data = data.split()

                    try: data = [int(x) for x in data]
                    except ValueError: pass  # For non-integer data, e.g. the LP.

                    if len(data) == 1:  # Removing unnecessary lists, e.g. for number of vehicles.
                        data = data[0]
                        
                    if self.prop_names[prop] in labels:  # For when there are multiple vehicles in the image.
                        labels[self.prop_names[prop]].append(data)
                    else:
                        labels[self.prop_names[prop]] = [data]

                elif "char" in prop:  # "char" for all the LP characters.
                    LP_chars_pos.append([int(x) for x in data.split(" ")])

                if print_log:
                    print(prop, "->", data)
                    
            
            labels["LP_chars_bb"] = LP_chars_pos
            labels["img_file_name"] = img_file_names[i]

            if not return_dict:
                all_labels.append(labels)
            else:
                if img_file_names[i] in all_labels:
                    print("Sample overidden...")
                all_labels[img_file_names[i]] = labels
        
        return all_labels


    def visualise_dataset(self, dataset_i=0, all_labels=None, subset=""):
        """ Plots all the images with all of its labels displayed. Use letters 'a' and 'd' to change images.
        
        Parameters:
        dataset_i: The dataset index, chosen from the class datasets variable.
        all_labels: All the labels for the images, excracted using the get_all_labels() method (default None).
        subset: If a dataset has a fixed train, val, test sets, specify which subset, "train", "val", "test" (default "").
        """

        if all_labels is None:
            all_labels = self.get_all_labels(dataset_i, print_log=False, subset=subset)

            if all_labels is None:
                return None

        path_to_imgs = f"{self.datasets[dataset_i]}/{subset}"  # Path to the direcotry where all the images are stored.
        
        img_index = 0  # 1201
        plt_sample_done = False  # Just to show an example on this jupyter notebook.
        
        while True:
            sample_info = all_labels[img_index]
            
            img_file_name = sample_info["img_file_name"]
            img_path = f"{path_to_imgs}/{img_file_name}"
            img = cv2.imread(img_path)
            
            # Image file name
            cv2.putText(img, f"file name: {img_file_name}", (30, 50), 0, 0.7, (0, 255, 0), 2)
            
            # Image number
            cv2.putText(img, f"#{img_index+1}", (30, 25), 0, 0.7, (0, 255, 0), 2)
            
            # Number of vehicles in the image.
            try:
                num_vehicles = sample_info["num_vehicles"][0]
            except KeyError:  # When num of vehicles was not specified in the annotations, in that case num of vechiles was one.
                num_vehicles = 1
            cv2.putText(img, f"# vehicles: {num_vehicles}", (30, 75), 0, 0.7, (0, 255, 0), 2)
            
            for i in range(num_vehicles):
                # Vehicle bounding box.
                v_bb = sample_info["v_bb"][i]
                x, y = v_bb[0], v_bb[1]
                w, h = v_bb[2], v_bb[3]
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Vehicle type
                vehicle_type = sample_info["v_type"][i]
                cv2.putText(img, vehicle_type, (x+10, y+h-10), 0, 0.6, (255, 255, 255), 2)

                # LP bounding box.
                LP_bb = sample_info["LP_bb"][i]
                x, y = LP_bb[0], LP_bb[1]
                w, h = LP_bb[2], LP_bb[3]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # LP text
                LP_chars = sample_info["LP_chars"][i]
                cv2.putText(img, str(LP_chars), (x, y-10), 0, 0.5, (0, 0, 255), 2)

            # All LP characters.
            LP_chars_bb = sample_info["LP_chars_bb"]
            bb_index = 0
            sample_info["LP_chars"] = [str(i) for i in sample_info["LP_chars"]]  # Some LP chars are all numbers.
            for LP_chars in sample_info["LP_chars"]:
                for LP_char in LP_chars:
                    char_bb = LP_chars_bb[bb_index]
                    bb_index += 1

                    # Char bb
                    x, y = char_bb[0], char_bb[1]
                    w, h = char_bb[2], char_bb[3]
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)

                    # Char text
                    cv2.putText(img, str(LP_char), (x, y+h+h), 2, 0.5, (255, 0, 255), 1)
            
            if not plt_sample_done:
                fig=plt.figure(figsize=(15, 15))
                plt.imshow(img[:, :, ::-1])
                plt_sample_done = True
                
            cv2.imshow("Image with labels", img)
            
            key = cv2.waitKey(1)
            
            if key == ord("q"): break
                
            elif key == ord("a"):
                if img_index != 0:
                    img_index -= 1
                
            elif key == ord("d"):
                if img_index != len(all_labels) - 1:
                    img_index += 1
                
        cv2.destroyAllWindows()


    def get_all_subset_labels(self, dataset_i, return_dict=False):
        """ Returns all labels of the dataset regardless if it is split into fixed train, val, test sets or not.
        
        If the dataset is split, all subsets will be combined as one dataset.

        Parameters:
        dataset_i: Dataset index.

        Returns:
        list: All sample annotations/labels for the dataset.
        """

        if dataset_i not in self.split_datasets_i:  # If dataset is not split into train, val, test sets.
            dataset_labels = self.get_all_labels(dataset_i, return_dict=return_dict)
        else:
            dataset_labels_train = self.get_all_labels(dataset_i=dataset_i, print_log=0, subset="training", return_dict=return_dict)
            dataset_labels_val = self.get_all_labels(dataset_i=dataset_i, print_log=0, subset="validation", return_dict=return_dict)
            dataset_labels_test = self.get_all_labels(dataset_i=dataset_i, print_log=0, subset="testing", return_dict=return_dict)

            if type(dataset_labels_train) is not dict:
                dataset_labels = dataset_labels_train + dataset_labels_val + dataset_labels_test
            else:
                dataset_labels = dataset_labels_train
                dataset_labels.update(dataset_labels_val)
                dataset_labels.update(dataset_labels_test)

        return dataset_labels


    def save_all_annos_as_dict(self):

        all_annos = {}

        for i, dataset in enumerate(self.datasets):
            annotations = self.get_all_subset_labels(i, return_dict=True)
            all_annos[dataset] = annotations

        file_name = "all_annos.json"

        with open(file_name, 'w') as file:
            json.dump(all_annos, file)

        print(f"All annotations saved in {file_name}.")

    def get_num_samples(self, dataset_i):
        """ Returns the number of samples for the dataset.
        
        Parameters:
        dataset_i: Dataset index.

        Returns:
        int: Number of samples in the dataset.
        """

        dataset_labels = self.get_all_subset_labels(dataset_i)
        return len(dataset_labels)

    def get_avg_num_veh(self, dataset_i):
        """ Returns the average # of vehicles in a sample.
        
        Parameters:
        dataset_i: Dataset index.

        Returns:
        float: Average # of vehicles in a sample.
        """

        dataset_labels = self.get_all_subset_labels(dataset_i)

        total_num_veh = 0
        for sample in dataset_labels:

            try:
                total_num_veh += sample["num_vehicles"][0]
            except KeyError: # When num of vehicles was not specified in the annotations, which means num vehicles=1.
                total_num_veh += 1

        avg_num_veh = total_num_veh / len(dataset_labels)

        return avg_num_veh

    
    def get_char_count(self, dataset_i):
        """ Returns the number of occurances each character appears in the LP across all dataset samples.

        Parameters:
        dataset_i: Dataset index.

        Returns:
        Counter: Each key is a character with the value being the number of occurances.
        """

        dataset_labels = self.get_all_subset_labels(dataset_i)

        # A string that holds all LPs of all vehicles in the dataset, which is then used to get number of occurances for each character.
        licence_plate_chars = ""

        for sample in dataset_labels:
            try:
                num_veh = sample["num_vehicles"][0]
            except KeyError: # When num of vehicles was not specified in the annotations, which means num vehicles=1.
                num_veh = 1
            
            for i in range(num_veh):
                licence_plate_chars += str(sample["LP_chars"][i])

        
        return Counter(licence_plate_chars)

    
    def check_tf_yolov4_annos(self, imgs_prefix, class_names_file, anno_file):
        """ Given YOLOv4 TF annotation compatible files, allows to visualise the annotations.

        Parameters:
        imgs_prefix: The path to all the images.
        class_names_file: The file containing all class names, where each class is on one line.
        anno_file: The YOLOv4 annotation file.
        """


        with open(class_names_file, "r") as file:
            names = file.read().split("\n")

        with open(anno_file, "r") as file:
            samples = file.read().split("\n")
            
            for i, line in enumerate(samples):
                samples[i] = line.split(" ")
                
        
        sample_i = 0

        while True:
            s1 = samples[sample_i]

            img_path = f"{imgs_prefix}{s1[0]}"

            # print(img_path)
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[0], img.shape[1]
            
            for i in range(1, len(s1)):
                sample_values = s1[i].split(",")
                c = sample_values[0]
                x, y = float(sample_values[1]), float(sample_values[2])
                w, h = float(sample_values[3]), float(sample_values[4])
                # print(i, sample_values, c, x, y, w, h)

                class_name = names[int(c)]
                
                bb_w, bb_h = w * img_w, h * img_h
                bb_x, bb_y = (x * img_w) - (bb_w/2), (y * img_h) - (bb_h/2)
                
                cv2.putText(img, class_name, (int(bb_x), int(bb_y - 5)), 1, 1, 255, 2)
                
                # print(bb_x, bb_y, bb_w, bb_h)
                img = cv2.rectangle(img, (int(bb_x), int(bb_y)), (int(bb_x) + int(bb_w), int(bb_y) + int(bb_h)), (255,0,0), 2)
                
            
            cv2.imshow("img", img)
            
            key = cv2.waitKey(1)

            # To do: Add constraint to sample_i.
            if key == ord("q"):
                break
            elif key == ord("a"):
                sample_i -= 1
            elif key == ord("d"):
                sample_i += 1
                
        cv2.destroyAllWindows()

    def check_darknet_yolov4_annos(self, dataset_dir, class_names_file, imgs_filetype="jpg"):
        """ Given YOLOv4 darknet annotation compatible files, allows to visualise the annotations.

        Parameters:
        dataset_dir: Path to the directory where the images and their corresponding .txt annotation files are located.
        class_names_file: The file containing all class names, where each class is on one line.
        imgs_filetype: The images file extension/file type (default "jpg").
        """

        filenames = os.listdir(dataset_dir)  # Includes both .imgs_filetype and .txt files.

        if len(filenames) == 0:
            print("No files found.")
            return

        with open(class_names_file, "r") as file:
            names = file.read().split("\n")
                
        
        sample_i = 0

        while True:
            img_filename = filenames[sample_i]
            anno_filename = filenames[sample_i + 1]

            img_path = f"{dataset_dir}/{img_filename}"
            anno_path = f"{dataset_dir}/{anno_filename}"

            img = cv2.imread(img_path)
            img_h, img_w = img.shape[0], img.shape[1]

            with open(anno_path, "r") as file:
                anno_lines = file.read().split("\n")

            for anno_line in anno_lines:
                if anno_line == "": continue
                sample_values = anno_line.split(" ")
                c_i = int(sample_values[0])  # class index
                x, y = float(sample_values[1]), float(sample_values[2])
                w, h = float(sample_values[3]), float(sample_values[4])

                class_name = names[c_i]

                bb_w, bb_h = w * img_w, h * img_h
                bb_x, bb_y = (x * img_w) - (bb_w/2), (y * img_h) - (bb_h/2)

                img = cv2.putText(img, class_name, (int(bb_x), int(bb_y - 5)), 1, 1, 255, 2)
                img = cv2.rectangle(img, (int(bb_x), int(bb_y)), (int(bb_x) + int(bb_w), int(bb_y) + int(bb_h)), (255,0,0), 2)

            cv2.imshow("LP patch", img)
            
            key = cv2.waitKey(1)

            # To do: Add constraint to sample_i.
            if key == ord("q"):
                break
            elif key == ord("a"):
                sample_i -= 2
            elif key == ord("d"):
                sample_i += 2
                
        cv2.destroyAllWindows()
    

    def generate_lp_det_data(self, dataset_i, save_dir_path, subset="", prefix="", darknet=True):
        """ Crops the vehicles from the images, saves them in save_dir_path, and creates LP BBs detection annotation file(s) (.txt).

        Parameters:
        dataset_i: Dataset index.
        save_dir_path: The root folder where the cropped vehicle images to be saved in.
        subset: If a dataset has a fixed train, val, test sets, specify which subset, "train", "val", "test" (default "").
        prefix: Prefix for images directory (default "").
        darknet: Whether to generate the annotations for the darknet YOLOv4 framework, if False, then for TF framework (default True).

        Returns:
        str: Annotation file path.
        """


        if subset == "":
            dataset_imgs_dir = self.datasets[dataset_i]
        else:  # For when a dataset has a fixed train, val, and test sets.
            dataset_imgs_dir = f"{self.datasets[dataset_i]}/{subset}"

        dataset_save_dir_path = f"{save_dir_path}/{dataset_imgs_dir}"
        # print(dataset_save_dir_path)

        if not os.path.exists(dataset_save_dir_path):
            os.makedirs(dataset_save_dir_path)

        dataset_labels = self.get_all_labels(dataset_i, subset=subset, prefix=prefix)

        if dataset_labels is None:  # For when a dataset has fixed train, val, and test sets.
            dataset_labels = self.generate_lp_det_data(dataset_i, save_dir_path, subset="training", prefix=prefix, darknet=darknet)
            dataset_labels = self.generate_lp_det_data(dataset_i, save_dir_path, subset="validation", prefix=prefix, darknet=darknet)
            dataset_labels = self.generate_lp_det_data(dataset_i, save_dir_path, subset="testing", prefix=prefix, darknet=darknet)
            return

        anno_str = ""  # Will hold final annotation file content(string).
        
        for sample in dataset_labels:

            try: num_vehicles = sample["num_vehicles"][0]
            except KeyError: num_vehicles = 1  # When no annotation is found, # of vehicles is one.

            img_filename = sample["img_file_name"]
            img_filepath = f"{prefix}{dataset_imgs_dir}/{img_filename}"
            
            img = cv2.imread(img_filepath)
            # plt.imshow(img)
            # plt.show()

            for i in range(num_vehicles):
                v_bb = sample["v_bb"][i]
                v_x, v_y = v_bb[0], v_bb[1]
                v_w, v_h = v_bb[2], v_bb[3]

                # Cropping vehicle and saving it as a new image.
                cropped_vehicle = img[v_y:v_y+v_h, v_x:v_x+v_w]
                sub_img_filepath = f"{dataset_save_dir_path}/{i}_{img_filename}"
                cv2.imwrite(sub_img_filepath, cropped_vehicle)
 

                lp_bb = sample["LP_bb"][i]
                lp_x, lp_y = lp_bb[0], lp_bb[1]
                lp_w, lp_h = lp_bb[2], lp_bb[3]


                # Getting the LP coordinates/size relative to the cropped vehicle patch.
                rel_lp_x = lp_x-v_x
                rel_lp_y = lp_y-v_y
                lp_patch = cropped_vehicle[rel_lp_y:rel_lp_y + lp_h, rel_lp_x:rel_lp_x + lp_w]

                lp_centre_x = (rel_lp_x + (lp_w / 2)) / v_w
                lp_centre_y = (rel_lp_y + (lp_h / 2)) / v_h
                lp_rel_w = lp_w / v_w
                lp_rel_h = lp_h / v_h

                if darknet:
                    anno_line = f"{self.lp_label} {lp_centre_x} {lp_centre_y} {lp_rel_w} {lp_rel_h}"
                    split_img_filename = img_filename.split(".")
                    anno_file_name = f"{split_img_filename[0]}.txt"
                    with open(f"{dataset_save_dir_path}/{i}_{anno_file_name}", "w") as file:
                        file.write(anno_line)
                    

                else:
                    # Adding the LP as one line in the annotation file.
                    # anno_line = f"{img_filepath}_{i} 0,{lp_centre_x},{lp_centre_y},{lp_rel_w},{lp_rel_h}\n"
                    anno_line = f"{sub_img_filepath} {self.lp_label},{lp_centre_x},{lp_centre_y},{lp_rel_w},{lp_rel_h}\n"
                    anno_str += anno_line

                    # plt.imshow(lp_patch)
                    # plt.show()

        # For the TensorFlow YOLOv4 framework.
        if not darknet:
            # Saving the annotations to file.
            anno_file_path = f"{dataset_save_dir_path}/{self.datasets[dataset_i]}_yolo_anno.txt"
            with open(anno_file_path, "w") as file:
                file.write(anno_str)
            
            return anno_file_path


    def split_dataset(self, path_to_anno_file, train=70, val=20):
        """ Splits the given annotation file into train, val, test sets using the percentages given, test=whats remaining.

        Paremeters:
        path_to_anno_file: Path to the .txt file which contains the annotations for each sample.
        train: Training set percentage (default 70).
        val: Validation set percentage (default 20).

        Returns:
        str: File paths to the .txt files for each set.
        """


        train_anno_file = ""
        val_anno_file = ""
        test_anno_file = ""

        with open(path_to_anno_file, "r") as file:
            lines = file.read().split("\n")

        n_samples = len(lines)

        train_n_samples = int(n_samples * (train/100))
        val_n_samples = int(n_samples * (val/100))

        # Splitting up the full annotation file(all lines) into seperate sets.
        train_lines = lines[:train_n_samples]
        val_lines = lines[train_n_samples:train_n_samples+val_n_samples]
        test_lines = lines[train_n_samples+val_n_samples:]

        # Adding the corresponding line to each annotation file string.
        for line in train_lines: train_anno_file += f"{line}\n"
        for line in val_lines: val_anno_file += f"{line}\n"
        for line in test_lines: test_anno_file += f"{line}\n"


        dirs_to_anno_file = path_to_anno_file.split("/")

        dir_to_directory = ""
        for i in range(len(dirs_to_anno_file) - 1):
            dir_to_directory += dirs_to_anno_file[i] + "/"

        
        train_set_path = f"{dir_to_directory}train.txt"
        val_set_path = f"{dir_to_directory}val.txt"
        test_set_path = f"{dir_to_directory}test.txt"
        
        with open(train_set_path, "w") as file: file.write(train_anno_file)
        with open(val_set_path, "w") as file: file.write(val_anno_file)
        with open(test_set_path, "w") as file: file.write(test_anno_file)

        return train_set_path, val_set_path, test_set_path
    

    def combine_anno_files(self, anno_file_paths, save_path, file_name):
        """ Combines the given annotation files into one single annotation file.

        Parameters:
        anno_file_paths: A list of all the annotation file paths.
        save_path: Where to save the combined annotation file.
        file_name: The file name for the combined annotation file.

        Returns:
        str: The combined annotation file path.
        """

        combined_annos_str = ""

        for anno_file_path in anno_file_paths:
            with open(anno_file_path, "r") as file:
                lines = file.read().split("\n")
            
            for line in lines:
                if line is not "" and line is not None:
                    combined_annos_str += line + "\n"
        
        file_path = f"{save_path}{file_name}"
        with open(file_path, "w") as file:
            file.write(combined_annos_str)
        
        return file_path

    def remove_txt_files(self, file_names):
        """ Removes .txt files from the given file_names list.

        Paremeters:
        file_names: A list of file_names.

        Returns:
        list: A list of all file names given with all .txt files removed from the list.

        """

        for file_name in file_names:
            if ".txt" in file_name:
                file_names.remove(file_name)

        return file_names
    
    def gen_dataset_split_files(self, dataset_i, imgs_root, imgs_prefix, train=70, val=20, seed=2):
        """ Generates train.txt, val.txt, test.txt for the given dataset, and saves them in save_dir, to be used with YOLOv4.

        Parameters:
        dataset_i: Dataset index, or for COCO dataset, it will be the name.
        imgs_root: The root directory for all the images for all datasets, not only for the given dataset.
        imgs_prefix: Prefix for each image file name, (the relative path to the split .txt files from wherever they will be used from).
        train: Training set percentage (default 70).
        val: Validation set percentage (default 20).
        seed: The seed used to shuffle the samples before the split is happened (default 2).

        """

        if type(dataset_i) != str:  # For the ALPR datasets.
            dataset_name = self.datasets[dataset_i]
            dataset_path = f"{imgs_root}/{dataset_name}"
            img_file_names = os.listdir(f"{dataset_path}")
        else:  # For coco downloaded dataset.
            dataset_name = dataset_i
            dataset_path = f"{imgs_root}/{dataset_name}"
            img_file_names = os.listdir(f"{dataset_path}")

        # For when the dataset is split into fixed train, val, and test sets, in that case, img_file_names will return
        # ["training", "validation", "testing"].
        if len(img_file_names) == 3 and "training" in img_file_names:  # The second condition is for reassurance.

            set_file_paths = ""  # For print summary at the end.

            for set_name in img_file_names:
                new_img_filenames = os.listdir(f"{dataset_path}/{set_name}")
                new_img_filenames = self.remove_txt_files(new_img_filenames)

                set_imgs_file = ""
                for img_filename in new_img_filenames: set_imgs_file += f"{imgs_prefix}/{dataset_name}/{set_name}/{img_filename}\n"

                subset_acro = ""
                if set_name == "training": subset_acro = "train"
                elif set_name == "validation": subset_acro = "val"
                else: subset_acro = "test"

                set_file = f"{imgs_root}/{subset_acro}_{dataset_name}.txt"
                set_file_paths += set_file + "\n"
                with open(set_file, "w") as file: file.write(set_imgs_file)

            print(f"{dataset_name} dataset done, files saved:\n{set_file_paths}")
            return  

        # Removing all .txt files.
        img_file_names = self.remove_txt_files(img_file_names)
        
        
        # Splitting up the imgs file names into the seperate sets.
        random.seed(seed)  # To make experiments consistent.
        random.shuffle(img_file_names)  # Ensuring no order is carried forward.

        n_samples = len(img_file_names)

        train_n_samples = int(n_samples * (train/100))
        val_n_samples = int(n_samples * (val/100))

        train_imgs = img_file_names[:train_n_samples]
        val_imgs = img_file_names[train_n_samples:train_n_samples+val_n_samples]
        test_imgs = img_file_names[train_n_samples+val_n_samples:]

        train_anno_file = ""
        val_anno_file = ""
        test_anno_file = ""

        for train_img in train_imgs: train_anno_file += f"{imgs_prefix}/{dataset_name}/{train_img}\n"
        for val_img in val_imgs: val_anno_file += f"{imgs_prefix}/{dataset_name}/{val_img}\n"
        for test_img in test_imgs: test_anno_file += f"{imgs_prefix}/{dataset_name}/{test_img}\n"


        train_set_path = f"{imgs_root}/train_{dataset_name}.txt"
        val_set_path = f"{imgs_root}/val_{dataset_name}.txt"
        test_set_path = f"{imgs_root}/test_{dataset_name}.txt"
        
        with open(train_set_path, "w") as file: file.write(train_anno_file)
        with open(val_set_path, "w") as file: file.write(val_anno_file)
        with open(test_set_path, "w") as file: file.write(test_anno_file)


        print(f"{dataset_name} dataset done, files saved:\n{train_set_path}\n{val_set_path}\n{test_set_path}\n")

    
    def combine_subsets(self, sets_dir_path):
        """ Combines train, val, and test sets of all datasets into three large subsets.

        Parameters:
        sets_dir_path: Where the individual subset .txt files are contained for all datasets. 
        """

        file_paths = glob.glob(f"{sets_dir_path}/*.txt")

        if file_paths == []:
            print("No subset files found, run gen_dataset_split_files() first.")
            return

        train_files = []
        val_files = []
        test_files = []

        for file_path in file_paths:
            file_name = file_path.split("\\")[-1]

            if "train_" in file_name: train_files.append(file_name)
            elif "val_" in file_name: val_files.append(file_name)
            elif "test_" in file_name: test_files.append(file_name)

        all_train_img_paths = []
        all_val_img_paths = []
        all_test_img_paths = []

        # Knowing that there is equal number of train, val, and test sets.
        for i in range(len(train_files)):

            with open(f"{sets_dir_path}/{train_files[i]}", "r") as file:
                train_paths = file.read().split("\n")[:-1]  # Split into lines and removes the last element which is an empty line break.
                all_train_img_paths += train_paths
            
            with open(f"{sets_dir_path}/{val_files[i]}", "r") as file:
                val_paths = file.read().split("\n")[:-1]
                all_val_img_paths += val_paths

            with open(f"{sets_dir_path}/{test_files[i]}", "r") as file:
                test_paths = file.read().split("\n")[:-1]
                all_test_img_paths += test_paths


        all_train_img_paths = "\n".join(all_train_img_paths)  # Converts lines to a string.
        all_val_img_paths = "\n".join(all_val_img_paths)
        all_test_img_paths = "\n".join(all_test_img_paths)

        all_train_path = f"{sets_dir_path}/all_train.txt"
        all_val_path = f"{sets_dir_path}/all_val.txt"
        all_test_path = f"{sets_dir_path}/all_test.txt"

        with open(all_train_path, "w") as file:
            file.write(all_train_img_paths)
        
        with open(all_val_path, "w") as file:
            file.write(all_val_img_paths)
        
        with open(all_test_path, "w") as file:
            file.write(all_test_img_paths)

        
        print(f"All subsets combined successfully, files_saved:\n{all_train_path}\n{all_val_path}\n{all_test_path}")

    def lp_patch_preprocessing(self, lp_patch):
        """ Used to apply preprocessing to LP patches to double the dataset for LP recognition.

        Parameters:
        lp_patch: The image of the LP patch.
        """

        # To be experimented with.

        return lp_patch

    
    def generate_lp_rec_data(self, dataset_i, save_dir_path, subset="", prefix="", negative_imgs=False, darknet=True):
        """ Crops the LPs from the vehicle cropped images, saves them in save_dir_path, and creates LP chars BBs detection annotation file(s) (.txt).

        Parameters:
        dataset_i: Dataset index.
        save_dir_path: The root folder where the cropped LP images to be saved in.
        subset: If a dataset has a fixed train, val, test sets, specify which subset, "train", "val", "test" (default "").
        prefix: Prefix for images directory (default "").
        negative_imgs: Whether to create negative images of each LP patch and save it as an additional sample. This will double the # of samples.
        darknet: Whether to generate the annotations for the darknet YOLOv4 framework, if False, then for TF framework (default True).

        Returns:
        str: Annotation file path.
        """


        if subset == "":
            dataset_imgs_dir = self.datasets[dataset_i]
        else:  # For when a dataset has a fixed train, val, and test sets.
            dataset_imgs_dir = f"{self.datasets[dataset_i]}/{subset}"

        dataset_save_dir_path = f"{save_dir_path}/{dataset_imgs_dir}"
        # print(dataset_save_dir_path)

        if not os.path.exists(dataset_save_dir_path):
            os.makedirs(dataset_save_dir_path)

        dataset_labels = self.get_all_labels(dataset_i, subset=subset, prefix=prefix)

        if dataset_labels is None:  # For when a dataset has fixed train, val, and test sets.
            dataset_labels = self.generate_lp_rec_data(dataset_i, save_dir_path, subset="training", prefix=prefix, negative_imgs=negative_imgs, darknet=darknet)
            dataset_labels = self.generate_lp_rec_data(dataset_i, save_dir_path, subset="validation", prefix=prefix, negative_imgs=negative_imgs, darknet=darknet)
            dataset_labels = self.generate_lp_rec_data(dataset_i, save_dir_path, subset="testing", prefix=prefix, negative_imgs=negative_imgs, darknet=darknet)
            return

        anno_str = ""  # Will hold final annotation file content(string).
        
        for sample in dataset_labels:

            # print(sample)

            try: num_vehicles = sample["num_vehicles"][0]
            except KeyError: num_vehicles = 1  # When no annotation is found, # of vehicles is one.

            img_filename = sample["img_file_name"]
            img_filepath = f"{prefix}{dataset_imgs_dir}/{img_filename}"
            
            img = cv2.imread(img_filepath)

            # plt.imshow(img)
            # plt.show()

            char_i_offset = 0

            for i in range(num_vehicles):
                v_bb = sample["v_bb"][i]
                v_x, v_y = v_bb[0], v_bb[1]
                v_w, v_h = v_bb[2], v_bb[3]

                # Cropped vehicle
                cropped_vehicle = img[v_y:v_y+v_h, v_x:v_x+v_w]
 

                lp_bb = sample["LP_bb"][i]
                lp_x, lp_y = lp_bb[0], lp_bb[1]
                lp_w, lp_h = lp_bb[2], lp_bb[3]


                # Getting the LP coordinates/size relative to the cropped vehicle patch and saving it.
                # rel_lp_x = lp_x-v_x
                # rel_lp_y = lp_y-v_y

                # lp_patch = cropped_vehicle[rel_lp_y:rel_lp_y + lp_h, rel_lp_x:rel_lp_x + lp_w]
                lp_patch = img[lp_y:lp_y+lp_h, lp_x:lp_x+lp_w]
                lp_patch = self.lp_patch_preprocessing(lp_patch)

                sub_img_filepath = f"{dataset_save_dir_path}/{i}_{img_filename}"
                cv2.imwrite(sub_img_filepath, lp_patch)

                if negative_imgs:
                    neg_lp_patch = ~lp_patch
                    neg_sub_img_filepath = f"{dataset_save_dir_path}/{i}_neg_{img_filename}"
                    cv2.imwrite(neg_sub_img_filepath, neg_lp_patch)

                lp_chars = str(sample["LP_chars"][i])

                anno_chars_str = ""  # Will hold the annotation file content, a line for each LP character.

                for char_i, char_bb in enumerate(sample["LP_chars_bb"]):
                    c_x, c_y = char_bb[0], char_bb[1]
                    c_w, c_h = char_bb[2], char_bb[3]

                    # Ensuring the each char belongs to the corresponding LP. If the char is not within the vehicle bb LP, skip char.
                    if not ((c_x >= lp_x) and (c_x <= (lp_x+lp_w)) and (c_y >= lp_y) and (c_y <= lp_y+lp_h)):
                        continue

                    rel_c_x = c_x - lp_x
                    rel_c_y = c_y - lp_y

                    c_centre_x = (rel_c_x + (c_w / 2)) / lp_w
                    c_centre_y = (rel_c_y + (c_h / 2)) / lp_h
                    c_rel_w = c_w / lp_w
                    c_rel_h = c_h / lp_h

                    current_char = str(lp_chars[char_i-char_i_offset])
                    char_label_index = self.lp_chars_labels.index(current_char)
                    if darknet:
                        anno_line = f"{char_label_index} {c_centre_x} {c_centre_y} {c_rel_w} {c_rel_h}"
                    else:
                        anno_line = f"{char_label_index},{c_centre_x},{c_centre_y},{c_rel_w},{c_rel_h}"

                    anno_chars_str += anno_line + "\n"

                char_i_offset += len(lp_chars)

                if darknet:
                    split_img_filename = img_filename.split(".")
                    anno_file_name = f"{i}_{split_img_filename[0]}.txt"

                    with open(f"{dataset_save_dir_path}/{anno_file_name}", "w") as file:
                        file.write(anno_chars_str)

                    # Creating the same annotation file for each negative sample.
                    if negative_imgs:
                        neg_anno_file_name = f"{i}_neg_{split_img_filename[0]}.txt"

                        with open(f"{dataset_save_dir_path}/{neg_anno_file_name}", "w") as file:
                            file.write(anno_chars_str)

                else:
                    anno_str += anno_chars_str
                
        # For the TensorFlow YOLOv4 framework.
        if not darknet:
            # Saving the annotations to file.
            anno_file_path = f"{dataset_save_dir_path}/{self.datasets[dataset_i]}_yolo_anno.txt"
            with open(anno_file_path, "w") as file:
                file.write(anno_str)
            
            return anno_file_path


    def generate_v_det_data(self, dataset_i, save_dir_path, subset="", prefix="", darknet=True):
        """ Creates vehicle BBs detection annotation files (.txt).

        Parameters:
        dataset_i: Dataset index.
        save_dir_path: The root folder where the vehicle images is to be saved in.
        subset: If a dataset has a fixed train, val, test sets, specify which subset, "train", "val", "test" (default "").
        prefix: Prefix for images directory (default "").
        darknet: Whether to generate the annotations for the darknet YOLOv4 framework, if False, then for TF framework (default True).

        Returns:
        str: Annotation file path.
        """


        if subset == "":
            dataset_imgs_dir = self.datasets[dataset_i]
        else:  # For when a dataset has a fixed train, val, and test sets.
            dataset_imgs_dir = f"{self.datasets[dataset_i]}/{subset}"

        dataset_save_dir_path = f"{save_dir_path}/{dataset_imgs_dir}"
        # print(dataset_save_dir_path)

        if not os.path.exists(dataset_save_dir_path):
            os.makedirs(dataset_save_dir_path)

        dataset_labels = self.get_all_labels(dataset_i, subset=subset, prefix=prefix)

        if dataset_labels is None:  # For when a dataset has fixed train, val, and test sets.
            dataset_labels = self.generate_v_det_data(dataset_i, save_dir_path, subset="training", prefix=prefix, darknet=darknet)
            dataset_labels = self.generate_v_det_data(dataset_i, save_dir_path, subset="validation", prefix=prefix, darknet=darknet)
            dataset_labels = self.generate_v_det_data(dataset_i, save_dir_path, subset="testing", prefix=prefix, darknet=darknet)
            return

        anno_str = ""  # Will hold final annotation file content(string).
        
        for sample in dataset_labels:

            try: num_vehicles = sample["num_vehicles"][0]
            except KeyError: num_vehicles = 1  # When no annotation is found, # of vehicles is one.

            img_filename = sample["img_file_name"]
            img_filepath = f"{prefix}{dataset_imgs_dir}/{img_filename}"
            
            img = cv2.imread(img_filepath)
            sub_img_filepath = f"{dataset_save_dir_path}/{img_filename}"
            cv2.imwrite(sub_img_filepath, img)
            # plt.imshow(img)
            # plt.show()

            sample_anno_file = ""  # Anno file content for a sample.

            for i in range(num_vehicles):
                v_bb = sample["v_bb"][i]
                v_x, v_y = v_bb[0], v_bb[1]
                v_w, v_h = v_bb[2], v_bb[3]
                v_type = sample["v_type"][i]

                img_h = img.shape[0]
                img_w = img.shape[1]

                v_centre_x = v_x + (v_w / 2)
                v_centre_y = v_y + (v_h / 2)
                rel_v_centre_x = v_centre_x / img_w
                rel_v_centre_y = v_centre_y / img_h
                rel_v_w = v_w / img_w
                rel_v_h = v_h / img_h

                if v_type == "car": v_class_id = 0
                elif v_type == "motorcycle": v_class_id = 1
                else:
                    print("Unkown class")
                    return

                if darknet:
                    anno_line = f"{v_class_id} {rel_v_centre_x} {rel_v_centre_y} {rel_v_w} {rel_v_h}\n"
                    sample_anno_file += anno_line
                    
                else:
                    anno_line = f"{sub_img_filepath} {v_class_id},{rel_v_centre_x},{rel_v_centre_y},{rel_v_w},{rel_v_h}\n"
                    anno_str += anno_line
            
            if darknet:
                split_img_filename = img_filename.split(".")
                anno_file_name = f"{split_img_filename[0]}.txt"
                with open(f"{dataset_save_dir_path}/{anno_file_name}", "w") as file:
                    file.write(sample_anno_file)

        # For the TensorFlow YOLOv4 framework.
        if not darknet:
            # Saving the annotations to file.
            anno_file_path = f"{dataset_save_dir_path}/{self.datasets[dataset_i]}_yolo_anno.txt"
            with open(anno_file_path, "w") as file:
                file.write(anno_str)
            
            return anno_file_path
    
    def download_coco(self, anno_file, class_names, dir_name, num_samples=5):
        """ Downloads given COCO class names samples and creates YOLO compatible annotation files.
        """

        try: coco = COCO(anno_file)
        except:
            print(f"Failed to load the coco annotation file \"{anno_file}\"")
            return
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        save_dir = f"coco/{dir_name}"
        try: os.makedirs(save_dir)
        except FileExistsError: pass

        class_ids = []
        images = []

        for class_name in class_names:
            print("Class:", class_name)
            catIds = coco.getCatIds(catNms=[class_name])[0]
            class_ids.append(catIds)
            print("Class ID:", catIds)

            imgIds = coco.getImgIds(catIds=catIds)
            print("Number of imgs:", len(imgIds))

            images += coco.loadImgs(imgIds)
            print("Example img:", images[-1], "\n")

        print("Class IDs", class_ids)
        total_imgs = len(images)
        print("Total imgs:", total_imgs, "(including duplicates)")

        imgs_done = os.listdir(save_dir)  # To prevent downloading twice next time this is ran.
        img_num = 0
        print_num = 100  # How many images to download to print an update.
        start_time = time.time()
        catIds = coco.getCatIds(catNms=class_names)

        for im in images:
            file_name = im['file_name']

            # To prevent "downloading" the same image twice.
            if file_name in imgs_done: continue

            try:
                img_data = requests.get(im['coco_url']).content
            except:
                print("Request broke, run again. All currently downloaded images will be skipped.")
                break

            with open(save_dir + '/' + im['file_name'], 'wb') as handler:
                handler.write(img_data)

            img_num += 1

            if img_num % print_num == 0:
                print(f"{img_num} images downloaded after {round((time.time() - start_time) / 60, 2)} minutes")


            dw = 1. / im['width']
            dh = 1. / im['height']
            
            annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
                
            filename = im['file_name'].replace(".jpg", ".txt")

            with open(f"{save_dir}/" + filename, "a") as myfile:
                for i in range(len(anns)):
                    class_id = anns[i]["category_id"]
                    xmin = anns[i]["bbox"][0]
                    ymin = anns[i]["bbox"][1]
                    xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
                    ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]
                    
                    x = (xmin + xmax)/2
                    y = (ymin + ymax)/2
                    
                    w = xmax - xmin
                    h = ymax - ymin
                    
                    x = x * dw
                    w = w * dw
                    y = y * dh
                    h = h * dh

                    class_label = class_ids.index(class_id)

                    # mystring = str(f"{class_label} " + str(truncate(x, 7)) + " " + str(truncate(y, 7)) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
                    mystring = f"{class_label} {x} {y} {w} {h}\n"
                    myfile.write(mystring)


            myfile.close()

            imgs_done += file_name

            if img_num == num_samples:
                print(f"Finished downloading {num_samples} images in", save_dir)
                break

        print("Finished downloading all images and creating annotation files for", class_names, "classes in\n", save_dir)



