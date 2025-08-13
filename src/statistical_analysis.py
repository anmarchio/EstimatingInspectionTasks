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


def compute_correlation_analysis(similarity_filepath, cross_results_dir):
    # --- Step 1: Load similarity matrix
    similarity_df = None
    with open(similarity_filepath, 'r', encoding='utf-8') as f:
        similarity_df = pd.read_csv(similarity_filepath, index_col=0,
                                    usecols=lambda col: col != similarity_filepath.split('/')[-1])

    if similarity_df is None:
        raise ValueError("No similarity data found!")

    # Normalize column and index names for matching
    similarity_df.columns = similarity_df.columns.astype(str).str.strip()
    similarity_df.index = similarity_df.index.astype(str).str.strip()

    # --- Step 2: Prepare to load cross-application results
    all_rows = []

    # Helper function to normalize dataset names from filenames
    def normalize_name(s):
        return s.replace("_mean_pipeline", "").replace(".txt", "").strip()

    # --- Step 3: Load each result file and collect scores
    response = requests.get(cross_results_dir)
    cross_results_files = [item for item in response.json() if item["type"] == "file"]

    for file in cross_results_files:
        #src_dataset = normalize_name(fname)
        #path = os.path.join(cross_results_dir, fname)

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

    # --- Step 5: Compute correlations
    pearson_corr, pearson_p = pearsonr(correlation_df["similarity"], correlation_df["cross_score"])
    spearman_corr, spearman_p = spearmanr(correlation_df["similarity"], correlation_df["cross_score"])

    print("✅ Pearson correlation:", pearson_corr, " (p =", pearson_p, ")")
    print("✅ Spearman correlation:", spearman_corr, " (p =", spearman_p, ")")

    # ------------ VISUALIZATION ----------
    sns.regplot(x='similarity', y='cross_score', data=correlation_df, scatter_kws={'alpha': 0.3})
    plt.title("Similarity vs Pipeline Transfer Performance")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Cross-Application Performance")
    plt.show()


def compute_linear_regression():
    result_dir = os.path.join(RESULTS_PATH)

    csv_files = [file for file in os.listdir(result_dir) if file.endswith(".csv")]

    for file in csv_files:
        similarity_df = read_df(file)
        X = similarity_df['similarity']
        y = similarity_df['performance']
        X = sm.add_constant(X)  # adds intercept
        model = sm.OLS(y, X).fit()
        print(model.summary())


def compute_mann_whitney_u():
    result_dir = os.path.join(RESULTS_PATH)

    csv_files = [file for file in os.listdir(result_dir) if file.endswith(".csv")]

    for file in csv_files:
        similarity_df = read_df(file)
        high = similarity_df[similarity_df['similarity'] > similarity_df['similarity'].quantile(0.66)]['performance']
        low = similarity_df[similarity_df['similarity'] < similarity_df['similarity'].quantile(0.33)]['performance']

        stat, p = mannwhitneyu(high, low, alternative='greater')  # one-sided
        print(f"Mann-Whitney U-test p-value (High vs Low similarity): {p:.5f}")
