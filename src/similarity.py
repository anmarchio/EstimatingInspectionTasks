import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from env_vars import DS_ROOT_PATH, IMG_SIZE
from experiment_params_data import DATASETS
from src.data_handling import write_df_to_csv
from src.models.entropy_metrics import jpeg_complexity, histogram_entropy, texture_features, edge_density, \
    number_of_superpixels, fourier_frequency
from src.models.resnet_embedding import resnet_embedding
from src.utils import print_similarity_matrix


def select_complexity_function():
    """Allow user to select a complexity function via command line."""
    print("\nSelect a complexity function:")
    print("[1] ResNet embedding")
    print("[2] JPEG Complexity")
    print("[3] Histrogram Entropy")
    print("[4] Texture Complexity")
    print("[5] Edge Density")
    print("[6] Number of Superpixels")
    print("[7] Fourier Frequency")
    print("[0] EXIT")

    choice = input("Enter your choice (1-7): ").strip()

    if "1" <= choice <= "7":
        compute_similarity(choice, list(DATASETS.keys()), [v['train'] for k, v in DATASETS.items()])
        return None
    else:
        print("Exiting similarity computation.")
        return None


def compute_similarity(choice, dataset_names, dataset_paths):
    all_embeddings = []
    for path in tqdm(dataset_paths):
        print("Processing:", path)
        full_path = os.path.join(DS_ROOT_PATH, path, "images")

        emb = None

        if choice == "1":
            emb = resnet_embedding(full_path)   # intentionally left untouched
        elif choice == "2":
            emb = jpeg_complexity(full_path)
        elif choice == "3":
            emb = histogram_entropy(full_path)
        elif choice == "4":
            emb = texture_features(full_path)
        elif choice == "5":
            emb = edge_density(full_path)
        elif choice == "6":
            emb = number_of_superpixels(full_path)
        elif choice == "7":
            emb = fourier_frequency(full_path)

        if emb is None:
            print(f"Warning: No images in {path}")
            emb = np.zeros((IMG_SIZE,))  # fallback
        all_embeddings.append(emb)

    # ---------- SIMILARITY MATRIX ----------
    all_embeddings = np.stack(all_embeddings)  # shape: (30, 2048)
    similarity_matrix = cosine_similarity(all_embeddings)  # shape: (30, 30)

    # ---------- SAVE TO CSV ----------
    df = pd.DataFrame(similarity_matrix, index=dataset_names, columns=dataset_names)

    result_path = write_df_to_csv(df)

    print_similarity_matrix(result_path)

    return result_path
