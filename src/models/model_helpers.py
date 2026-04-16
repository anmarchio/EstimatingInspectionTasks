import os

import cv2
import numpy as np


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_image_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    ])


def load_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def load_image_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def safe_mean_std(features):
    """
    Aggregate per-image features into one dataset embedding:
    [mean(features), std(features)]
    """
    features = np.asarray(features, dtype=np.float32)

    if features.ndim == 1:
        features = features[:, None]

    mean_vec = np.mean(features, axis=0)
    std_vec = np.std(features, axis=0)

    return np.concatenate([mean_vec, std_vec], axis=0)


def shannon_entropy_from_hist(hist):
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))