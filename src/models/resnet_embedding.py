import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from tqdm import tqdm

from env_vars import IMG_SIZE, DS_ROOT_PATH
from src.data_handling import write_df_to_csv


def get_model():
    # ---------- LOAD RESNET50 BASE ----------
    resnet_base = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    feature_extractor = Model(inputs=resnet_base.input, outputs=resnet_base.output)

    return feature_extractor


def load_and_preprocess_images(dataset_path, max_images=50):
    embeddings = []
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][
                  :max_images]

    for img_name in image_files:
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img)
        embeddings.append(img)

    if not embeddings:
        return None

    input_batch = np.stack(embeddings)
    feature_extractor = get_model()
    features = feature_extractor.predict(input_batch, verbose=0)
    return np.mean(features, axis=0)  # Mean embedding per dataset


def print_similarity_matrix(file_path):
    """Reads the similarity matrix from a CSV file and prints it."""
    try:
        df = pd.read_csv(file_path)
        similarity_matrix = df.values
        dataset_names = df.columns.tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    # ---------- DISPLAY RESULTS ----------
    print("\nCosine Similarity Matrix (rounded):\n")
    header = "     " + "  ".join([name[:5] for name in dataset_names])
    print(header)
    for i, name in enumerate(dataset_names):
        row = f"{name[:5]} " + "  ".join([f"{similarity_matrix[i, j]:.2f}" for j in range(len(dataset_names))])
        print(row)


def compute_similarity_matrix(dataset_names, dataset_paths):
    all_embeddings = []
    for path in tqdm(dataset_paths):
        print("Processing:", path)
        full_path = os.path.join(DS_ROOT_PATH, path, "images")
        emb = load_and_preprocess_images(full_path)
        if emb is None:
            print(f"Warning: No images in {path}")
            emb = np.zeros((IMG_SIZE,))  # fallback
        all_embeddings.append(emb)

    # ---------- SIMILARITY MATRIX ----------
    all_embeddings = np.stack(all_embeddings)  # shape: (30, 2048)
    similarity_matrix = cosine_similarity(all_embeddings)  # shape: (30, 30)

    # ---------- SAVE TO CSV ----------
    df = pd.DataFrame(similarity_matrix, index=dataset_names, columns=dataset_names)

    print_similarity_matrix(df)

    return write_df_to_csv(df)
