import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2
import glob
from skimage.feature import graycomatrix, graycoprops

from env_vars import DS_ROOT_PATH
from experiment_params_data import DATASETS


def extract_image_statistics(image):
    """Extracts statistical features from a grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Intensity features
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Edge density (Sobel filter)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge_density = np.mean(edges > 50)

    # Texture features using GLCM (Gray-Level Co-Occurrence Matrix)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    return [mean_intensity, std_intensity, edge_density, contrast, entropy]


def compute_cluster_similarities(features, labels):
    similarities = {}
    for cluster in np.unique(labels):
        indices = np.where(labels == cluster)[0]
        subset = features[indices]
        sim_matrix = cosine_similarity(subset)
        np.fill_diagonal(sim_matrix, np.nan)  # Ignore self-comparisons
        similarities[cluster] = sim_matrix
    return similarities


def select_pairs(similarities, labels):
    selected_pairs = []
    for cluster, sim_matrix in similarities.items():
        indices = np.where(labels == cluster)[0]
        for i, dataset_idx in enumerate(indices):
            sorted_sim = np.argsort(sim_matrix[i, :])  # Rank by similarity
            top_2 = sorted_sim[-3:-1]  # Top 2 similar
            medium_2 = sorted_sim[len(sorted_sim) // 3: len(sorted_sim) // 3 + 2]
            least_1 = sorted_sim[0]
            selected = np.concatenate([top_2, medium_2, [least_1]])
            selected_pairs.append((dataset_idx, indices[selected]))
    return selected_pairs


def plot_similarity_heatmap(similarities):
    for i, sim_matrix in enumerate(similarities):
        if similarities[i].ndim != 2:
            print(f"Cluster {i} is not a 2-d input: {similarities[i].shape}")
            continue
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarities[i], annot=True, cmap='coolwarm', cbar=True)
        plt.title(f'Similarity Heatmap for Cluster {i}')
        plt.show(block=True)


def generate_random_binary_mask(shape=(256, 256)):
    # Simulated ground truth & predicted segmentations (binary masks)
    return np.random.randint(0, 2, shape)


def compute_mcc_selected_pairs(selected_pairs):
    mcc_results = []
    for dataset_idx, test_datasets in selected_pairs:
        gt_mask = generate_random_binary_mask()
        for test_idx in test_datasets:
            pred_mask = generate_random_binary_mask()
            mcc = matthews_corrcoef(gt_mask.flatten(), pred_mask.flatten())
            mcc_results.append((dataset_idx, test_idx, mcc))

    print("Sample MCC scores:", mcc_results[:5])

    return mcc_results


def bootstrap_mcc(mcc_values, num_samples=10000):
    sampled_mcc = np.random.choice(mcc_values, (num_samples, len(mcc_values)), replace=True)
    boot_means = np.mean(sampled_mcc, axis=1)
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    return np.mean(boot_means), ci_lower, ci_upper


def compute_anova_and_correlation(mcc_results):
    # Organize MCC values by similarity group
    high_sim_mcc = [m[2] for m in mcc_results[:50]]
    med_sim_mcc = [m[2] for m in mcc_results[50:100]]
    low_sim_mcc = [m[2] for m in mcc_results[100:150]]

    # ANOVA test
    f_stat, p_val = stats.f_oneway(high_sim_mcc, med_sim_mcc, low_sim_mcc)
    print(f"ANOVA test: F={f_stat:.3f}, p={p_val:.3f}")

    # Pearson correlation
    similarity_scores = np.random.uniform(0, 1, 150)  # Simulated similarities
    mcc_values = np.array([m[2] for m in mcc_results])
    corr, p_corr = stats.pearsonr(similarity_scores, mcc_values)

    print(f"Pearson Correlation: r={corr:.3f}, p={p_corr:.3f}")


def cluster_datasets(dataset_features, dataset_names):
    # Determine the optimal number of clusters (Elbow Method)
    inertia = []
    K_range = range(2, 10 if len(dataset_features) >= 10 else len(dataset_features))

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(dataset_features)
        inertia.append(kmeans.inertia_)

    plt.plot(K_range, inertia, 'bo-')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title("Elbow Method for Optimal K")
    plt.show(block=True)

    # Cluster using the optimal K (e.g., k=6 based on elbow)
    optimal_k = 6

    manual_k = input("Enter the optimal K value: ")
    try:
        optimal_k = int(manual_k)
    except ValueError:
        print("Invalid input. Using default K=6.")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(dataset_features)

    # Visualizing clusters using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(dataset_features)

    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.title("Dataset Clusters in Reduced Feature Space")
    plt.show(block=True)

    return cluster_labels


def compute_similarity_within_clusters(features, labels):
    similarities = {}
    for cluster in np.unique(labels):
        indices = np.where(labels == cluster)[0]
        subset = features[indices]
        sim_matrix = cosine_similarity(subset)
        np.fill_diagonal(sim_matrix, np.nan)  # Ignore self-comparisons
        similarities[cluster] = sim_matrix
    return similarities


def main():
    # ------------------------------------------------
    # Extract features for each dataset (assume we have folders for each dataset)
    # ------------------------------------------------
    dataset_features = []
    dataset_names = []
    for k in DATASETS.keys():
        folder = os.path.join(DS_ROOT_PATH, DATASETS[k]['train'], "images")
        images = [cv2.imread(img) for img in
                  glob.glob(f"{folder}/*.[jJ][pP][gG]") + glob.glob(f"{folder}/*.[pP][nN][gG]") + glob.glob(
                      f"{folder}/*.[bB][mM][pP]")[:10]]  # Sample 10 images per dataset
        if len(images) == 0:
            print(f"No images found in {folder}")
            continue
        dataset_features.append(np.mean([extract_image_statistics(img) for img in images], axis=0))
        dataset_names.append(k)

    dataset_features = np.array(dataset_features)
    print("Extracted Features Shape:", dataset_features.shape)  # (30, 5) if 30 datasets

    cluster_labels = cluster_datasets(dataset_features, dataset_names)

    # ------------------------------------------------
    # Compute similarity within clusters and select dataset pairs
    # ------------------------------------------------
    similarities = compute_similarity_within_clusters(dataset_features, cluster_labels)

    plot_similarity_heatmap(similarities)

    # ------------------------------------------------
    # Compute MCC scores for selected dataset pairs
    # ------------------------------------------------
    selected_pairs = select_pairs(similarities, cluster_labels)
    print("Selected dataset pairs:", selected_pairs)

    raise NotImplementedError
    mcc_results = compute_mcc_selected_pairs(selected_pairs)

    # ------------------------------------------------
    # Bootstrap MCC scores for statistical analysis
    # ------------------------------------------------
    mcc_values = [m[2] for m in mcc_results]
    mean_mcc, ci_low, ci_high = bootstrap_mcc(mcc_values)

    print(f"Bootstrap MCC mean: {mean_mcc:.3f} (95% CI: {ci_low:.3f} - {ci_high:.3f})")
    sns.histplot(mcc_values, kde=True)
    plt.title("Bootstrapped MCC Distribution")
    plt.show(block=True)

    # ------------------------------------------------
    # ANOVA and correlation analysis
    # ------------------------------------------------
    compute_anova_and_correlation(mcc_results)


if __name__ == "__main__":
    main()
