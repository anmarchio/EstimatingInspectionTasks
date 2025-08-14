import glob
import os

import cv2
import numpy as np
import requests
import scipy.stats as stats
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import spearmanr, pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

import statsmodels.api as sm
from env_vars import RESULTS_PATH, LONG_TO_SHORT_NAME
from src.data_handling import read_df


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


def normalize_name(s):
    # Helper function to normalize dataset names from filenames
    return s.replace("_mean_pipeline", "").replace(".txt", "").strip()


def load_and_prepare_similarity_and_cross_results(similarity_filepath, cross_results_dir):
    # --- Step 1: Load similarity matrix
    similarity_df = None
    with open(similarity_filepath, 'r', encoding='utf-8') as f:
        similarity_df = pd.read_csv(similarity_filepath, index_col=0,
                                    usecols=lambda col: col != similarity_filepath.split('/')[-1])

    if similarity_df is None:
        raise ValueError("No similarity data found!")

    # --- Step 2: Prepare to load cross-application results
    all_rows = []

    # Normalize column and index names for matching
    similarity_df.columns = similarity_df.columns.astype(str).str.strip()
    similarity_df.index = similarity_df.index.astype(str).str.strip()

    # --- Step 3: Load each result file and collect scores
    response = requests.get(cross_results_dir)
    cross_results_files = [item for item in response.json() if item["type"] == "file"]

    for file in cross_results_files:
        # src_dataset = normalize_name(fname)
        # path = os.path.join(cross_results_dir, fname)

        raw_url = file["download_url"]  # direct raw file link
        src_dataset = file["name"]

        if not src_dataset.endswith("_mean_pipeline.txt"):
            continue

        # Download file content directly into pandas
        df = pd.read_csv(raw_url, sep=';', engine='python')

        for _, row in df.iterrows():
            tgt_dataset = normalize_name(row[2])
            src_dataset = normalize_name(src_dataset)
            try:
                similarity = similarity_df.loc[LONG_TO_SHORT_NAME[src_dataset], LONG_TO_SHORT_NAME[tgt_dataset]]
            except KeyError:
                continue  # Skip pairs not found in similarity matrix

            all_rows.append({
                "source": src_dataset,
                "target": tgt_dataset,
                "similarity": similarity,
                "cross_score": float(row[3])
            })

    # --- Step 4: Create dataframe for analysis
    correlation_df = pd.DataFrame(all_rows)

    return similarity_df, correlation_df


def compute_correlation_analysis(similarity_filepath, cross_results_dir):
    _, correlation_df = load_and_prepare_similarity_and_cross_results(similarity_filepath, cross_results_dir)

    # --- Step 5: Compute correlations
    pearson_corr, pearson_p = pearsonr(correlation_df["similarity"], correlation_df["cross_score"])
    spearman_corr, spearman_p = spearmanr(correlation_df["similarity"], correlation_df["cross_score"])

    print("✅ Pearson correlation:", pearson_corr, " (p =", pearson_p, ")")
    print("✅ Spearman correlation:", spearman_corr, " (p =", spearman_p, ")")

    # --- Step 6: Compute correlations for different similarity ranges
    low_similarity = correlation_df[correlation_df["similarity"] < 0.5]
    medium_similarity = correlation_df[(correlation_df["similarity"] >= 0.5) & (correlation_df["similarity"] < 0.7)]
    high_similarity = correlation_df[correlation_df["similarity"] >= 0.7]

    # Low similarity
    low_pearson_corr, low_pearson_p = pearsonr(low_similarity["similarity"], low_similarity["cross_score"])
    low_spearman_corr, low_spearman_p = spearmanr(low_similarity["similarity"], low_similarity["cross_score"])
    print("✅ Low Similarity - Pearson correlation:", low_pearson_corr, " (p =", low_pearson_p, ")")
    print("✅ Low Similarity - Spearman correlation:", low_spearman_corr, " (p =", low_spearman_p, ")")

    # Medium similarity
    medium_pearson_corr, medium_pearson_p = pearsonr(medium_similarity["similarity"], medium_similarity["cross_score"])
    medium_spearman_corr, medium_spearman_p = spearmanr(medium_similarity["similarity"],
                                                        medium_similarity["cross_score"])
    print("✅ Medium Similarity - Pearson correlation:", medium_pearson_corr, " (p =", medium_pearson_p, ")")
    print("✅ Medium Similarity - Spearman correlation:", medium_spearman_corr, " (p =", medium_spearman_p, ")")

    # High similarity
    high_pearson_corr, high_pearson_p = pearsonr(high_similarity["similarity"], high_similarity["cross_score"])
    high_spearman_corr, high_spearman_p = spearmanr(high_similarity["similarity"], high_similarity["cross_score"])
    print("✅ High Similarity - Pearson correlation:", high_pearson_corr, " (p =", high_pearson_p, ")")
    print("✅ High Similarity - Spearman correlation:", high_spearman_corr, " (p =", high_spearman_p, ")")

    # ------------ VISUALIZATION ----------
    sns.regplot(x='similarity', y='cross_score', data=correlation_df, scatter_kws={'alpha': 0.3})
    plt.title("Similarity vs Pipeline Transfer Performance")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Cross-Application Performance")
    plt.show()


def compute_linear_regression(similarity_filepath, cross_results_dir):
    similarity_df, correlation_df = load_and_prepare_similarity_and_cross_results(similarity_filepath,
                                                                                  cross_results_dir)

    # --- Step 6: Compute linear regression for the entire dataset
    X = correlation_df['similarity']
    y = correlation_df['cross_score']
    X = sm.add_constant(X)  # adds intercept
    model = sm.OLS(y, X).fit()
    print(model.summary())


def compute_mann_whitney_u(similarity_filepath, cross_results_dir):
    similarity_df, correlation_df = load_and_prepare_similarity_and_cross_results(similarity_filepath,
                                                                                  cross_results_dir)

    # --- Step 6: Compute correlations for different similarity ranges
    low_similarity = correlation_df[correlation_df["similarity"] < 0.5]
    medium_similarity = correlation_df[(correlation_df["similarity"] >= 0.5) & (correlation_df["similarity"] < 0.7)]
    high_similarity = correlation_df[correlation_df["similarity"] >= 0.7]

    high = high_similarity["cross_score"]
    low = low_similarity["cross_score"]

    stat, p = mannwhitneyu(high, low, alternative='greater')  # one-sided
    print(f"Mann-Whitney U-test p-value (High vs Low similarity): {p:.5f}")
