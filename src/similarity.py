import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from env_vars import DS_ROOT_PATH, IMG_SIZE
from experiment_params_data import DATASETS
from src.data_handling import write_df_to_csv
from src.utils import print_similarity_matrix


def select_complexity_function():
    """Allow user to select a complexity function via command line."""
    print("\nSelect a complexity function:")
    print("[1] ResNet embedding")
    print("[2] Edge Complexity")
    print("[3] Gradient Complexity")
    print("[4] Texture Complexity")
    print("[5] Fourier Frequency")
    print("[0] EXIT")

    choice = input("Enter your choice (1-4): ").strip()

    if choice >= "1" and choice <= "5":
        compute_similarity(choice, list(DATASETS.keys()), [v['train'] for k, v in DATASETS.items()])
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
            emb = load_and_preprocess_images(full_path)
        elif choice == "2":
            emb = compute_edge()
        elif choice == "3":
            emb = compute_gradient()
        elif choice == "4":
            emb = compute_texture()
        elif choice == "5":
            emb = compute_frequency()

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

"""
Also see: https://stackoverflow.com/questions/50313114/what-is-the-entropy-of-an-image-and-how-is-it-calculated

Metrics selected for complexity analysis:
* Entropy: as defined by the shannon entropy
* Blurriness
* Brightness
* Img Size: size of the image in pixels `W x H`
* lbl Size: size of the label's contained pixels
* label_count_per_image: no. of labels per image
* relative_label_size: size of label in px compared to image
* hist_entropy: ?
* jpeg_complexity
* fractal_dimension
* texture_features
* edge_density
* laplacian_variance
* num_superpixels
"""
def compute_edge(image=None):
    """Compute edge complexity."""
    pass


def compute_gradient(image=None):
    """Compute gradient complexity."""
    pass


def compute_texture(image=None):
    """Compute texture complexity."""
    pass


def compute_frequency(image=None):
    """Compute Fourier frequency."""
    pass
