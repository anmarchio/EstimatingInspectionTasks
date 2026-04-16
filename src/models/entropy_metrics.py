import cv2
import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import slic

from src.models.model_helpers import list_image_files, load_image_gray, shannon_entropy_from_hist, safe_mean_std, \
    load_image_color


# -----------------------------
# 1) JPEG complexity
# -----------------------------
def jpeg_complexity(folder_path, jpeg_quality=75):
    """
    Measures how much each image compresses with JPEG.
    Feature per image:
        [compression_ratio, mse_after_jpeg]
    Dataset embedding:
        mean + std => length 4
    """
    image_files = list_image_files(folder_path)
    if len(image_files) == 0:
        return None

    feats = []

    for path in image_files:
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img, dtype=np.uint8)

            # Raw size estimate: width * height * channels
            raw_size = img_np.size  # number of scalar values

            # Encode to JPEG in memory
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            success, enc = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), encode_param)
            if not success:
                continue

            jpeg_size = len(enc)
            compression_ratio = raw_size / (jpeg_size + 1e-8)

            # Decode back and compute reconstruction error
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

            mse = np.mean((img_np.astype(np.float32) - dec.astype(np.float32)) ** 2)

            feats.append([compression_ratio, mse])

        except Exception:
            continue

    if len(feats) == 0:
        return None

    return safe_mean_std(feats)


# -----------------------------
# 2) Histogram entropy
# -----------------------------
def histogram_entropy(folder_path, bins=64):
    """
    Entropy-based intensity distribution features.
    Feature per image:
        [entropy, mean_intensity, std_intensity]
    Dataset embedding:
        mean + std => length 6
    """
    image_files = list_image_files(folder_path)
    if len(image_files) == 0:
        return None

    feats = []

    for path in image_files:
        img = load_image_gray(path)
        if img is None:
            continue

        hist = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
        entropy = shannon_entropy_from_hist(hist)

        mean_intensity = float(np.mean(img))
        std_intensity = float(np.std(img))

        feats.append([entropy, mean_intensity, std_intensity])

    if len(feats) == 0:
        return None

    return safe_mean_std(feats)


# -----------------------------
# 3) Texture features (GLCM)
# -----------------------------
def texture_features(folder_path, resize_to=(256, 256)):
    """
    Gray-Level Co-occurrence Matrix features.
    Feature per image:
        [contrast, dissimilarity, homogeneity, energy, correlation, ASM]
    Dataset embedding:
        mean + std => length 12
    """
    image_files = list_image_files(folder_path)
    if len(image_files) == 0:
        return None

    feats = []

    for path in image_files:
        img = load_image_gray(path)
        if img is None:
            continue

        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)

        # Reduce to fewer levels for stable GLCM
        img_q = (img // 32).astype(np.uint8)  # 8 levels: 0..7

        glcm = graycomatrix(
            img_q,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=8,
            symmetric=True,
            normed=True
        )

        contrast = graycoprops(glcm, "contrast").mean()
        dissimilarity = graycoprops(glcm, "dissimilarity").mean()
        homogeneity = graycoprops(glcm, "homogeneity").mean()
        energy = graycoprops(glcm, "energy").mean()
        correlation = graycoprops(glcm, "correlation").mean()
        asm = graycoprops(glcm, "ASM").mean()

        feats.append([contrast, dissimilarity, homogeneity, energy, correlation, asm])

    if len(feats) == 0:
        return None

    return safe_mean_std(feats)


# -----------------------------
# 4) Edge density
# -----------------------------
def edge_density(folder_path, resize_to=(256, 256)):
    """
    Edge-based features using Canny.
    Feature per image:
        [edge_ratio, edge_mean, edge_std]
    Dataset embedding:
        mean + std => length 6
    """
    image_files = list_image_files(folder_path)
    if len(image_files) == 0:
        return None

    feats = []

    for path in image_files:
        img = load_image_gray(path)
        if img is None:
            continue

        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(img, threshold1=100, threshold2=200)

        edge_ratio = float(np.mean(edges > 0))
        edge_mean = float(np.mean(edges))
        edge_std = float(np.std(edges))

        feats.append([edge_ratio, edge_mean, edge_std])

    if len(feats) == 0:
        return None

    return safe_mean_std(feats)


# -----------------------------
# 5) Number of superpixels
# -----------------------------
def number_of_superpixels(folder_path, resize_to=(256, 256), n_segments=200, compactness=10):
    """
    SLIC superpixel statistics.
    Since SLIC always aims at n_segments, we use the actual produced count
    and average superpixel size statistics.

    Feature per image:
        [actual_num_segments, mean_segment_size, std_segment_size]
    Dataset embedding:
        mean + std => length 6
    """
    image_files = list_image_files(folder_path)
    if len(image_files) == 0:
        return None

    feats = []

    for path in image_files:
        img = load_image_color(path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)

        try:
            segments = slic(
                img,
                n_segments=n_segments,
                compactness=compactness,
                start_label=0
            )

            labels, counts = np.unique(segments, return_counts=True)
            actual_num_segments = len(labels)
            mean_segment_size = float(np.mean(counts))
            std_segment_size = float(np.std(counts))

            feats.append([actual_num_segments, mean_segment_size, std_segment_size])

        except Exception:
            continue

    if len(feats) == 0:
        return None

    return safe_mean_std(feats)


# -----------------------------
# 6) Fourier frequency
# -----------------------------
def fourier_frequency(folder_path, resize_to=(256, 256)):
    """
    Frequency-domain features from FFT magnitude spectrum.
    Feature per image:
        [low_band_energy, mid_band_energy, high_band_energy, high_low_ratio]
    Dataset embedding:
        mean + std => length 8
    """
    image_files = list_image_files(folder_path)
    if len(image_files) == 0:
        return None

    feats = []

    for path in image_files:
        img = load_image_gray(path)
        if img is None:
            continue

        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)

        h, w = mag.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        r_max = np.max(r)
        low_mask = r <= 0.15 * r_max
        mid_mask = (r > 0.15 * r_max) & (r <= 0.4 * r_max)
        high_mask = r > 0.4 * r_max

        low_energy = float(np.mean(mag[low_mask]))
        mid_energy = float(np.mean(mag[mid_mask]))
        high_energy = float(np.mean(mag[high_mask]))
        high_low_ratio = high_energy / (low_energy + 1e-8)

        feats.append([low_energy, mid_energy, high_energy, high_low_ratio])

    if len(feats) == 0:
        return None

    return safe_mean_std(feats)
