import glob
import os

import cv2
import numpy as np
import scipy.stats as stats
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import spearmanr, pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

import statsmodels.api as sm
from env_vars import RESULTS_PATH
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


def compute_correlation_analysis():
    result_dir = os.path.join(RESULTS_PATH)

    csv_files = [file for file in os.listdir(result_dir) if file.endswith(".csv")]

    for file in csv_files:
        similarity_df = read_df(file)

        # Assuming df is your DataFrame with columns: similarity, performance
        spearman_corr, spearman_p = spearmanr(similarity_df['similarity'], similarity_df['performance'])
        pearson_corr, pearson_p = pearsonr(similarity_df['similarity'], similarity_df['performance'])

        print(f"Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.5f})")
        print(f"Pearson correlation:  {pearson_corr:.3f} (p={pearson_p:.5f})")

        # ------------ VISUALIZATION ----------
        sns.regplot(x='similarity', y='performance', data=similarity_df, scatter_kws={'alpha': 0.3})
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
