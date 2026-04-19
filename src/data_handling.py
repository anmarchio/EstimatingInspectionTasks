import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from keras.src.utils import img_to_array
from env_vars import IMG_SIZE, RESULTS_PATH
from src.utils import get_dataset_identifier, log_message


def read_df(file_path):
    print(f"\nReading results from {file_path} ...")
    similarity_df = None
    try:
        similarity_df = pd.read_csv(file_path, index_col=0)
        print(similarity_df.to_string())
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return similarity_df


def write_df_to_csv(df, category:str = "", target_dir:str = None):
    if target_dir is None:
        target_dir = os.path.join(RESULTS_PATH)

    if category is not "":
        category = f"_{category}"

    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

    result_file = os.path.join(
        target_dir,
        f"{current_datetime}{category}.csv"
    )

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    df.to_csv(result_file, index=True)

    return result_file


def print_table(filepath: str):
    try:
        df = pd.read_csv(filepath)
        print(df.to_string(index=False))


    except Exception as e:
        print(f"Error reading CSV file: {e}")
        log_message("data_handling->print_table", get_dataset_identifier(filepath), str(e))


def load_data_with_labels(train_images: [], train_labels: [], mask_as_gray=True):
    try:
        images = []
        labels = []
        for i in range(len(train_images)):
            img = cv2.imread(train_images[i])
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)

            mask_array = None
            if mask_as_gray:
                # mask_array = imread(train_labels[i], as_gray=True)
                mask = cv2.imread(train_labels[i], cv2.IMREAD_GRAYSCALE)
                mask_array = mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            else:
                mask = cv2.imread(train_labels[i])
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                mask_array = img_to_array(mask) / 255.0
            labels.append(mask_array)
    except Exception as e:
        print(str(e))
        log_message("data_handling->load_data", get_dataset_identifier(train_images[0]), str(e))

    return np.array(images), np.array(labels)
