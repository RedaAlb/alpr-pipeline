import os
import random
import shutil
from collections import Counter

import Automold as am
import cv2
import matplotlib.pyplot as plt


class DataGenerator:
    """ Used to generate new samples using various methods.
    """

    def __init__(self):
        self.lp_chars_labels = ["0","1","2","3","4","5","6","7","8","9", "A","B","C","D","E","F","G","H",
                                "I","J", "K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]


    def get_lp_annotations(self, imgs_dir):
        """ Get all LP character annotations (img path, char id, x, y, w, h).

        Args:
            imgs_dir (str): Path to the directory where all the images are located alongside their .txt annotation files.

        Returns:
            list: All annotations for all LPs in the given image directory.
        """

        annos = []
        
        filenames = os.listdir(imgs_dir)
        filenames = [x for x in filenames if x.split(".")[-1] != "txt"]  # Removing all .txt files from the list.

        for filename in filenames:
            img = cv2.imread(f"{imgs_dir}/{filename}")
            img_h, img_w, _ = img.shape
            
            
            # Excrating the annotations for each character in the LP.
            lp_annotations = []  # Will hold all the annotations for all character of the LP.
            
            img_name = filename.split(".")[0]
            anno_filename = img_name + ".txt"
            
            with open(f"{imgs_dir}/{anno_filename}", "r") as file:
                anno_lines = file.read().split("\n")
                
                char_annotations = []
                
                # Each line is a character with data (char_id, x, y, w, h) seperated by spaces.
                for line in anno_lines:
                    if line == "": continue  # Skip empty lines.
                        
                    char_data = line.split(" ")
                    char_id = char_data[0]
                    
                    # Converting the coordinates to top left corner of each character rather then the centre.
                    bb_w, bb_h = float(char_data[3]) * img_w, float(char_data[4]) * img_h
                    bb_x, bb_y = float(char_data[1]) * img_w - (bb_w/2), float(char_data[2]) * img_h - (bb_h/2)
                    
                    # Storing the annos as [img_path, char_id, x, y, w, h].
                    char_anno = [char_id, int(bb_x), int(bb_y), int(bb_w), int(bb_h)]
                    char_annotations.append(char_anno)
                
                lp_annotations.append(char_annotations)
                lp_annotations.insert(0, f"{imgs_dir}/{filename}")
            
            annos.append(lp_annotations)
        
        return annos


    def get_chars_occ_from_imgs(self, imgs_dir, low_ap_letters):
        """ Get the occurances of all characters from images and their .txt annotation files.

        Args:
            imgs_dir (str): Path to the directory where all the images are located alongside their .txt annotation files.
            low_ap_letters (list): The low performing (low average precision) letters.

        Returns:
            Counter: Number of occurances (occ) for each character, where the key is the character and value is the num of occ.
        """

        samples_data = self.get_lp_annotations(imgs_dir)
        n_samples = len(samples_data)
        
        digits_count = []
        letters_count = []
        l_aps_count = []  # Number of low ap chars in each LP.
        all_chars = ""
        
        for sample in samples_data:
            img_path = sample[0]
            
            chars = sample[1]

            # Seperating the LP characters into digits, letters and the low ap letters.
            digits = []
            letters = []
            l_ap_letters = []
            
            for char_data in chars:
                class_id = int(char_data[0])
                char = self.lp_chars_labels[class_id]
                all_chars += char

                if class_id < 10:  # Digits
                    digits.append(char_data)
                else:
                    if char in low_ap_letters:
                        l_ap_letters.append(char_data)
                    else:    
                        letters.append(char_data)
            
            l_ap_letters = l_ap_letters * 2
            
            digits_count.append(len(digits))
            letters_count.append(len(letters))
            l_aps_count.append(len(l_ap_letters))

        
        print(imgs_dir)
        total_digits = sum(digits_count)
        total_letters = sum(letters_count)
        total_l_aps = sum(l_aps_count)

        print("Average number of digits in an LP:", total_digits/n_samples, ", total:", total_digits)
        print("Average number of letters in an LP:", total_letters/n_samples, ", total:", total_letters)
        print("Average number of low AP letters in an LP:", total_l_aps/n_samples, ", total:", total_l_aps)
        print("Total characters in dataset:", sum([total_digits, total_letters, total_l_aps]))
        print("\n")

        return Counter(all_chars)


    def gen_permutations(self,
                         imgs_dir,
                         save_dir_name,
                         low_ap_letters,
                         low_ap_dupl=1,
                         lp_all_letters=False,
                         samples_to_display=-1,
                         exclude_a=False,
                         replace_1=True,
                         only_low_ap=False,
                         save_org=True):
        """ Generate permutations.

        Args:
            imgs_dir (str): Path to the directory where all the images are located alongside their .txt annotation files.
            save_dir_name (str): Saving directory for the newly generated data.
            low_ap_letters (list): The low performing (low average precision) letters.
            low_ap_dupl (int, optional): How many times a low AP character is allowed to duplicate in the same LP. Defaults to 1.
            lp_all_letters (bool, optional): Whether to make the whole LP just letters, so replace all digits in the LP. Defaults to False.
            samples_to_display (int, optional): How many samples to display of the generated data, use -1 for no display. Defaults to -1.
            exclude_a (bool, optional): Exclude the character "a", since it appears significantly more than other chars. Defaults to False.
            replace_1 (bool, optional): Whether to replace the digit "1" as it is very narrow and distorts most letters. Defaults to True.
            only_low_ap (bool, optional): Make the LP only made up of low AP characters. Defaults to False.
            save_org (bool, optional): Whether to save the original LP patch as a seperate sample. Defaults to True.
        """

        samples_data = self.get_lp_annotations(imgs_dir)
        
        dataset_dir = "/".join(imgs_dir.split("/")[1:])
        full_save_dir = f"{save_dir_name}/{dataset_dir}"
        try: os.makedirs(full_save_dir)
        except FileExistsError: pass
        
        
        if samples_to_display != -1:
            _, ax = plt.subplots(samples_to_display, 2, figsize=(10, 2.5 * samples_to_display))
        
        for i, sample in enumerate(samples_data):
            img_path = sample[0]
            img = cv2.imread(img_path)
            
            if samples_to_display != -1:
                ax[i][0].imshow(img)
                ax[i][0].set_title("Original")
            
            chars = sample[1]
            a_digit_replaced = False

            if save_org:
                self.save_sample(img, img_path, chars, full_save_dir)
            
            # These will store the digits, all chars, and the low ap chars in the LP in seperate list.
            digits = []
            letters = []
            l_ap_letters = []
            
            # Seperating the LP characters into digits, letters and the low ap letters.
            for char in chars:
                class_id = int(char[0])
                
                if class_id < (10 + exclude_a):  # Digits
                    if replace_1:
                        digits.append(char)
                    else:
                        if class_id != 1:
                            digits.append(char)

                else:
                    letter = self.lp_chars_labels[class_id]
                    if letter in low_ap_letters:
                        l_ap_letters.append(char)
                    elif not only_low_ap:
                        letters.append(char)
            
            l_ap_letters = l_ap_letters * low_ap_dupl
            
            backup_letters = letters[:]
            backup_ap_letters = l_ap_letters[:]
            
            
            for digit in digits:
                letter = None
                if len(l_ap_letters) != 0:
                    letter = l_ap_letters.pop()
                elif len(letters) != 0:
                    letter = letters.pop()
                    
                    if lp_all_letters:  # Making the whole LP characters letters, keep replacing digits until no digits left.
                        # When letters is empty, it means both l_ap_letters and letters are empty.
                        # So resetting both to go again when all l_ap_letters and letters are exhausted.
                        if len(letters) == 0:
                            letters = backup_letters[:]
                            l_ap_letters = backup_ap_letters[:]
                
                if letter is not None:
                    # Replacing the digit with the letter.
                    img = self.replace_digit(img, digit, letter)
                    a_digit_replaced = True
                    
                    # Ensuring the label for the digit is changed to the letter.
                    digit_index = chars.index(digit)
                    chars[digit_index] = [letter[0], digit[1], digit[2], digit[3], digit[4]]
            
            if a_digit_replaced:
                self.save_sample(img, img_path, chars, full_save_dir, img_prefix="gen_")

                    
            if samples_to_display != -1:
                ax[i][1].imshow(img)
                ax[i][1].set_title("Auto generated")
                
                if i+1 == samples_to_display:
                    break


    def replace_digit(self, img, digit, letter):
        """ Replace a digit of an LP patch with a letter.

        Args:
            img (numpy.ndarray): The LP patch, where the digit and character are in.
            digit (list): The digit patch bounding box info in this format [x, y, w, h], based on the top left corner.
            letter (list): The letter patch bounding box info in this format [x, y, w, h], based on the top left corner.

        Returns:
            numpy.ndarray: The same passed in img, but with the digit patch replaced by the letter patch.
        """

        d_x, d_y = digit[1], digit[2]
        d_w, d_h = digit[3], digit[4]

        l_x, l_y = letter[1], letter[2]
        l_w, l_h = letter[3], letter[4]

        digit_patch = img[d_y:d_y+d_h, d_x:d_x+d_w]
        d_h, d_w, _ = digit_patch.shape

        letter_patch = img[l_y:l_y+l_h, l_x:l_x+l_w]
        
        # Resizing the letter patch to match the digit patch.
        letter_patch = cv2.resize(letter_patch, (d_w, d_h))

        # Replacing the digit patch with the letter patch in the original image.
        img[d_y:d_y+d_h, d_x:d_x+d_w] = letter_patch
        
        return img


    def save_sample(self, img, img_path, annos, save_dir, img_prefix=""):
        """ Save a sample with its annotation file.

        Args:
            img (numpy.ndarray): The image to be saved.
            img_path (str): The image path.
            annos (list): The annotations for the image in this format [class id, x, y, w, h].
            save_dir (str): Saving directory for the sample.
            img_prefix (str, optional): A prefix before the saved image filename. Defaults to "".
        """

        img_h, img_w, _ = img.shape
        
        anno_lines = []
        
        for anno in annos:
            class_id = anno[0]
            x, y = anno[1], anno[2]
            w, h = anno[3], anno[4]
            
            # Converting coordinates to centre of BB and relative to img width and height.
            centre_x, centre_y = x + (w/2), y + (h/2)
            
            rel_x, rel_y = centre_x / img_w, centre_y / img_h
            rel_w, rel_h = w / img_w, h / img_h
            
            anno_line = f"{class_id} {rel_x} {rel_y} {rel_w} {rel_h}"
            anno_lines.append(anno_line)
        
        filename = img_path.split("/")[-1].split(".")[0]
        save_path = f"{save_dir}/{img_prefix}{filename}"
        
        cv2.imwrite(f"{save_path}.jpg", img)
        
        with open(f"{save_path}.txt", "w") as file:
            file_content = "\n".join(anno_lines)
            file.write(file_content)


    def get_low_ap_paths(self, root_dir, path_to_imgs, low_ap_letters):
        """ Get all the image paths that contain the low average precision (AP) letters.

        Args:
            root_dir (dir): The root directory of all the datasets.
            path_to_imgs (list): The path to the images per dataset from the root_dir.
            low_ap_letters (list): The low performing (low AP) letters.

        Returns:
            list: All the low AP image paths.
        """

        low_ap_img_paths = []

        for path in path_to_imgs:
            annos = self.get_lp_annotations(f"{root_dir}/{path}")

            dataset_paths = []

            for anno in annos:
                img_path = anno[0]
                lp_chars = anno[1]

                for lp_char in lp_chars:
                    char = self.lp_chars_labels[int(lp_char[0])]
                    if char in low_ap_letters:
                        dataset_paths.append(img_path)
                        break

            print(path, "- number of low AP characters:", len(dataset_paths))
            low_ap_img_paths.append(dataset_paths)
        
        return low_ap_img_paths


    def gen_rand_aug_imgs(self, root_dir, path_to_imgs, low_ap_letters, output_dir):
        """ Generate random augmentation images by adding shadow, redish colour, or blur to the images.

        Args:
            root_dir (dir): The root directory of all the datasets.
            path_to_imgs (list): The path to the images per dataset from the root_dir.
            low_ap_letters (list): The low performing (low AP) letters.
            output_dir (str): The saving directory for the newly generated data.
        """

        low_ap_img_paths = self.get_low_ap_paths(root_dir, path_to_imgs, low_ap_letters)

        for dataset_paths in low_ap_img_paths:
            for path in dataset_paths:        
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                
                rand = random.randint(1, 100)
                
                # Each augmentation having 33% chance of being applied.
                if rand < 33:
                    proc_img = am.add_shadow(img, no_of_shadows=4, rectangular_roi=(-1,-1,-1,-1), shadow_dimension=4)
                elif rand >= 33 and rand < 66:
                    proc_img = am.add_autumn(img)
                else:
                    proc_img = am.m_add_blur(img, low_kernel=3, high_kernel=5, w_thresh=150)[0]
                            
                
                dataset_name = path.split("/")[1]
                new_path = f"{output_dir}/{dataset_name}"
                
                try: os.makedirs(new_path)
                except FileExistsError: pass
                
                img_filename = path.split("/")[-1]
                img_name = img_filename.split(".")[0]
                img_ext = img_filename.split(".")[-1]
                
                new_img_path = f"{output_dir}/{dataset_name}/{img_name}_gen.{img_ext}"
                plt.imsave(new_img_path, proc_img)

                txt_filepath = "/".join(path.split("/")[:-1]) + "/" + img_name + ".txt"
                new_txt_filepath = f"{output_dir}/{dataset_name}/{img_name}_gen.txt"
                shutil.copy(txt_filepath, new_txt_filepath)

        print("\nRandom data augmentation generation done.")
