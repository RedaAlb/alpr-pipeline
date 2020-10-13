import cv2
import os

from collections import Counter

class DatasetsUtil:
    """ Provides utility functions to operate on the datasets."""

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


    # TODO: Possible rename to get_all_samples, and all references to "labels".
    def get_all_labels(self, dataset_i=0, print_log=False, subset=""):
        """ Excract the needed labels from the annotation (anno) files.
        
        Parameters:
        dataset_i: The dataset index to get the labels to. Chosen from the class datasets variable.
        print_log: Whether to print the structure of the lines and values (default False).
        subset: If a dataset has a fixed train, val, test sets, specify which subset, "train", "val", "test" (default "").
        
        Returns:
        list: All the labels for all the samples in the dataset, where each element is a dict for each sample, with key names
        according to prop_names, which also includes added keys LP_chars_bb and img_file_name.
        """

        # Please note, all paths are considered relative to this notebook.
        path_to_anno = f"annotations/{self.datasets[dataset_i]}/{subset}"  # Path to the directory where all the annotations are stored.
        path_to_imgs = f"{self.datasets[dataset_i]}/{subset}"  # Path to the direcotry where all the images are stored.

        anno_file_names = os.listdir(path_to_anno)
        img_file_names = os.listdir(path_to_imgs)
        
        all_labels = []  # Will hold all labels for all the samples in the dataset, where each element is a dict.
        
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
            
            all_labels.append(labels)
        
        return all_labels


    def visualise_dataset(self, dataset_i=0, all_labels=None, subset=""):
        """ Plots all the images with all of its labels displayed. Use letters 'a' and 'd' to change images.
        
        Parameters:
        dataset_i: The dataset index to get the labels to. Chosen from the class datasets variable.
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


    def get_all_subset_labels(self, dataset_i):
        """ Returns all labels of the dataset regardless if it is split into fixed train, val, test sets or not.
        
        If the dataset is split, all subsets will be combined as one dataset.

        Parameters:
        dataset_i: Dataset index.

        Returns:
        list: All sample annotations/labels for the dataset.
        """

        if dataset_i not in self.split_datasets_i:  # If dataset is not split into train, val, test sets.
            dataset_labels = self.get_all_labels(dataset_i)
        else:
            dataset_labels_train = self.get_all_labels(dataset_i=dataset_i, print_log=0, subset="training")
            dataset_labels_val = self.get_all_labels(dataset_i=dataset_i, print_log=0, subset="validation")
            dataset_labels_test = self.get_all_labels(dataset_i=dataset_i, print_log=0, subset="testing")
            dataset_labels = dataset_labels_train + dataset_labels_val + dataset_labels_test

        return dataset_labels


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






