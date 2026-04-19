import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from env_vars import DS_ROOT_PATH, IMG_SIZE, RESULTS_PATH, SIMILARITY_DIR
from experiment_params_data import DATASETS
from src.data_handling import write_df_to_csv
from src.models.entropy_metrics import jpeg_complexity, histogram_entropy, texture_features, edge_density, \
    number_of_superpixels, fourier_frequency
from src.models.resnet_embedding import resnet_embedding
from src.utils import print_similarity_matrix

actions = {
    1: "resnet_embedding",
    2: "jpeg_complexity",
    3: "histogram_entropy",
    4: "texture_complexity",
    5: "edge_density",
    6: "no_of_superpixels",
    7: "fourier_frequency",
}

def compute_complexity_metrics():
    """
    Run all available complexity-to-similarity computations sequentially.
    For each metric we print a short action label before invoking compute_similarity.
    """

    dataset_names = list(DATASETS.keys())
    dataset_paths = [v['train'] for k, v in DATASETS.items()]

    target_dir = os.path.join(SIMILARITY_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    result_paths = []

    for i in range(1, len(actions.keys())):
        label = actions.get(i, f"Metric {i}")
        print(f"Computing {label}...")
        result_paths.append(compute_similarity(i, target_dir, dataset_names, dataset_paths))

    return result_paths


def print_similarity_distribution(result_path):
    try:
        df = pd.read_csv(result_path, index_col=0)
    except Exception as e:
        print(f"Error reading file ` {result_path} `: {e}")
        return

    mat = df.values

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        print(f"Unexpected matrix shape: {mat.shape}")
        return

    n = mat.shape[0]
    if n < 2:
        print("Not enough elements for computing distribution.")
        return

    # take upper triangle (exclude diagonal)
    vals = mat[np.triu_indices(n, k=1)]
    total = vals.size

    # Define thresholds and labels (from low to high)
    thresholds = {
        'None': (0.0, 0.3),
        'Low': (0.5, 0.5),
        'Medium': (0.5, 0.7),
        'High': (0.7, 1.0)
    }

    counts = {}
    for label, (low, high) in thresholds.items():
        if label == 'None':
            mask = np.isclose(vals, 0.0)
        else:
            # low exclusive, high inclusive for Low/Medium/High
            mask = (vals > low) & (vals <= high)
        counts[label] = int(mask.sum())

    # Print header and ranges
    print("| Similarity | Range                | Count | Percentage |")
    for label in ['None', 'Low', 'Medium', 'High']:
        cnt = counts[label]
        pct = (cnt / total * 100) if total > 0 else 0.0
        low, high = thresholds[label]
        if label == None:
            range_str = "None"
        else:
            range_str = f"> {low} and <= {high}"
        print(f"| {label:<6} | {range_str:<18} | {cnt:5d} | {pct:9.2f}% |")

    # Sanity-check: ensure counts sum to total (if not, print remainder as 'Other')
    counted = sum(counts.values())
    if counted != total:
        other = int(total - counted)
        other_pct = (other / total * 100) if total > 0 else 0.0
        print(f"| Other  | {'(unclassified)':<18} | {other:5d} | {other_pct:9.2f}% |")


def compute_similarity(choice, target_dir, dataset_names, dataset_paths):
    """
    :param choice:
        1: ResNet embedding
        2: JPEG Complexity
        3: Histogram Entropy
        4: Texture Complexity
        5: Edge Density
        6: Number of Superpixels
        7: Fourier Frequency
    :param dataset_names:

    :param dataset_paths:
    :return:
    """
    all_embeddings = []
    for path in tqdm(dataset_paths):
        print("Processing: ", path)
        full_path = os.path.join(DS_ROOT_PATH, path, "images")

        emb = None

        if choice == 1:
            print("Skipping ResNet embedding for ", path)
            #emb = resnet_embedding(full_path)
        elif choice == 2:
            emb = jpeg_complexity(full_path)
        elif choice == 3:
            emb = histogram_entropy(full_path)
        elif choice == 4:
            emb = texture_features(full_path)
        elif choice == 5:
            emb = edge_density(full_path)
        elif choice == 6:
            emb = number_of_superpixels(full_path)
        elif choice == 7:
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

    result_path = write_df_to_csv(df,
                                  category=f"{actions[choice]}",
                                  target_dir=target_dir)

    print_similarity_matrix(result_path)
    
    print_similarity_distribution(result_path)
    
    return result_path
