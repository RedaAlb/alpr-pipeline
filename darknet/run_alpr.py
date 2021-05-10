from alpr_pipeline import ALPRPipeline


# Vehicle detection
# A dict with the required files and values for the darknet detector in this format:
# (.cfg, .data, .weights, detection threshold, minimum patch width in pixels, min patch height):
V_DET_INFO = {"cfg": "../../../../saved_data/v_det.cfg",
              "data": "../../../../saved_data/v_det.data",
              "weights": "../../../../saved_data/v_det.weights",
              "thresh": 0.25,
              "min_patch_w": 40,
              "min_patch_h": 40}

# LP detection
LP_DET_INFO = {"cfg": "../../../../saved_data/lp_det.cfg",
              "data": "../../../../saved_data/lp_det.data",
              "weights": "../../../../saved_data/lp_det.weights",
              "thresh": 0.65,
              "min_patch_w": 20,
              "min_patch_h": 10}

# Character detection/LP recognition
CHAR_DET_INFO = {"cfg": "../../../../saved_data/lp_rec.cfg",
                 "data": "../../../../saved_data/lp_rec.data",
                 "weights": "../../../../saved_data/lp_rec.weights",
                 "thresh": 0.75,
                 "min_patch_w": 2,
                 "min_patch_h": 2}


# Vehicle type classification (vtc)
VTC_MODEL_PATH = "../../../../../vtc/saved_models/vtc_final_model.h5"
VTC_CLASS_NAMES = ["emergency", "other", "truck"]



VIDEO = False  # Whether to use video or images. If camera needed, specify the which camera source input to use.
VIDEO_PATH = "video.mp4"

# If VIDEO is false, then images will be taken from this img file paths .txt file. Use the "d" and "a" keys to navigate samples.
IMGS_PATH_TXT = "data/all_test_full_imgs.txt"
ALL_ANNOS_FILE = "data/all_annos.json"  # Get this json file from using the DatasetsUtils.save_all_annos_as_dict()


DISPLAY_DETECTIONS = True  # Display all detections (vehicle, LP, characters) on the frame.
DISPLAY_ORG_FRAME = False  # Display the original frame without any BBs.
DISPLAY_V_PATCH = False    # Display the cropped vehicle patches.
DISPLAY_LP_PATCH = False   # Display the cropped LP patches.
DISPLAY_CHARS_DET = False  # Display the individual character detections/BBs.


EVALUATE = False  # Whether to evaluate and record key metrics such as recall and wrong samples for each stage in the pipeline.
PRINT_SAMPLE_EVAL = False  # Whether to print evaluation for each sample/image.
SAVE_V_DET_WRONG = False  # Whether to save the wrong samples where the vehicle detection was wrong.
SAVE_LP_DET_WRONG = False
SAVE_LP_REC_WRONG = False


alpr_pipeline = ALPRPipeline(V_DET_INFO,
                            LP_DET_INFO,
                            CHAR_DET_INFO,
                            VTC_MODEL_PATH,
                            VTC_CLASS_NAMES,
                            VIDEO,
                            VIDEO_PATH,
                            IMGS_PATH_TXT,
                            ALL_ANNOS_FILE,
                            DISPLAY_DETECTIONS,
                            DISPLAY_ORG_FRAME,
                            DISPLAY_V_PATCH,
                            DISPLAY_LP_PATCH,
                            DISPLAY_CHARS_DET,
                            EVALUATE,
                            PRINT_SAMPLE_EVAL,
                            SAVE_V_DET_WRONG,
                            SAVE_LP_DET_WRONG,
                            SAVE_LP_REC_WRONG)


alpr_pipeline.run()