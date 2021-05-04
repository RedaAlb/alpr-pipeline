import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter


class DataGenerator:


    def __init__(self):
        self.lp_chars_labels = ["0","1","2","3","4","5","6","7","8","9", "A","B","C","D","E","F","G","H",
                                "I","J", "K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

    def get_annotations(self, imgs_dir):
    
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
        samples_data = self.get_annotations(imgs_dir)
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


        samples_data = self.get_annotations(imgs_dir)
        
        dataset_dir = "/".join(imgs_dir.split("/")[1:])
        full_save_dir = f"{save_dir_name}/{dataset_dir}"
        try: os.makedirs(full_save_dir)
        except FileExistsError: pass
        
        
        if samples_to_display != -1: _, ax = plt.subplots(samples_to_display, 2, figsize=(10, 2.5*samples_to_display))
        
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