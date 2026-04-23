import os
import string
from datetime import datetime
from random import random

import pandas as pd
import requests

import env_vars
from env_vars import SHORT_TO_LONG_NAME, LONG_TO_SHORT_NAME


def replace_ambigous_title(title_str: str):
    title_str = title_str.replace('_training', '')
    title_str = title_str.replace('_train', '')
    title_str = title_str.replace('/training', '')
    title_str = title_str.replace('training', '')
    title_str = title_str.replace('train', '')
    title_str = title_str.replace('train_cgp', '')
    title_str = title_str.replace('_cgp', '')
    title_str = title_str.replace('_large', '_lg')
    title_str = title_str.replace('_small', '_sm')
    title_str = title_str.replace('\"', '')
    return title_str


def get_dataset_identifier(dataset_path):
    # Split the path by the OS-specific separator
    parts = dataset_path.split(os.sep)
    del parts[-1]
    del parts[-1]

    parts[-1] = replace_ambigous_title(parts[-1])

    # Ensure no part is an empty string
    parts = [part for part in parts if part]

    # Determine how to construct the identifier
    if len(parts) > 3:
        # Combine the third and second-to-last parts for a more descriptive name
        identifier = f"{parts[-3]}_{parts[-1]}"
    elif len(parts) > 2:
        # Fall back to combining the second-to-last and last parts
        identifier = f"{parts[-2]}_{parts[-1]}"
    elif parts[-1] != "D:" and parts[-1] != "C:":
        identifier = parts[-1]
    else:
        # avoid empty identifier
        identifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    return identifier


def log_message(function: str, dataset: str, msg: str):
    logdir = os.path.join(os.getcwd(), "log")
    current_date = datetime.now().strftime("%Y%m%d")
    curlogfile = os.path.join(logdir, f"log{current_date}.txt")

    log_entry = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ", " \
                + msg + " in " \
                + function + " for dataset " \
                + dataset + "\n"

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    with open(curlogfile, "a") as logfile:
        logfile.write(log_entry)


# ---------- MAIN HELPER FUNCTIONS ----------

def select_dir(results_dir) -> []:
    try:
        dirs = [entry for entry in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, entry))]
    except FileNotFoundError:
        print(f"Empty DIR: {results_dir}.")
        return []

    print("Found the following directories:")
    for idx, file in enumerate(dirs):
        print(f"[{idx}] `{file}`")

    selection_idx = input("Select DIR index: ")
    try:
        selection_idx = int(selection_idx)
        if selection_idx < 0 or selection_idx >= len(dirs):
            print("Invalid selection.")
            return []

        chosen_dir = dirs[selection_idx]
        csv_files = [os.path.join(results_dir, chosen_dir, file)
                     for file in os.listdir(os.path.join(results_dir, chosen_dir))
                     if file.endswith(".csv")]

        if not csv_files:
            print(f"No CSV files found in {results_dir}.")
            return []

    except ValueError:
        print("Invalid selection.")
        return []

    return csv_files


def select_similarity_folder(base_similarity_dir):
    """Listet Unterordner in `base_similarity_dir` (z.B. timestamped folders) auf und lässt den User einen auswählen.
    Gibt den vollständigen Pfad zum gewählten Ordner zurück oder None, wenn abgebrochen/keine Ordner.
    """
    try:
        entries = [e for e in os.listdir(base_similarity_dir) if os.path.isdir(os.path.join(base_similarity_dir, e))]
    except Exception as e:
        print(f"Could not list similarity folders in {base_similarity_dir}: {e}")
        return None

    if not entries:
        print(f"No similarity subfolders found in {base_similarity_dir}.")
        return None

    print("Found similarity folders:")
    for idx, name in enumerate(sorted(entries)):
        print(f"[{idx}] {name}")

    sel = input("Select folder index (or press Enter to cancel): ")
    if sel.strip() == "":
        print("Cancelled selection.")
        return None

    try:
        sel_idx = int(sel)
    except ValueError:
        print("Invalid selection.")
        return None

    if sel_idx < 0 or sel_idx >= len(entries):
        print("Selection out of range.")
        return None

    chosen = sorted(entries)[sel_idx]
    return os.path.join(base_similarity_dir, chosen)


def build_similarity_files_from_dir(similarity_dir):
    """Scans `similarity_dir` nach CSV-Dateien und bildet ein dict mit erwarteten Tags.

    Erwartete Tags und typische Schlüsselwörter in Dateinamen:
      - 'cnn': 'resnet', 'cnn'
      - 'edge': 'edge', 'edgeDen'
      - 'texture': 'text', 'texture', 'textComp'
      - 'entropy': 'hist', 'entropy', 'histEnt', 'histogram'
      - 'frequency': 'four', 'freq', 'frequency', 'fourFreq'
      - 'superpixel': 'super', 'noOfSup', 'superpixel'

    Gibt ein dict zurück mit gefundenen Pfaden (nur für vorhandene Dateien).
    """
    if not os.path.isdir(similarity_dir):
        print(f"Not a directory: {similarity_dir}")
        return {}

    files = [f for f in os.listdir(similarity_dir) if f.lower().endswith('.csv')]
    files_lc = {f.lower(): f for f in files}

    # mapping tag -> list of candidate substrings
    patterns = {
        'cnn': ['resnet', 'cnn'],
        'edge': ['edge', 'edgeden', 'edgedensity', 'edgeden'],
        'texture': ['text', 'texture', 'textcomp', 'texturecomp'],
        'entropy': ['hist', 'histent', 'histogram', 'entropy'],
        'frequency': ['four', 'fourfreq', 'frequency', 'freq'],
        'superpixel': ['super', 'noofsup', 'superpixel']
    }

    found = {}

    # try to match each pattern to a filename
    for tag, keys in patterns.items():
        match = None
        for fname_lc, fname in files_lc.items():
            for key in keys:
                if key in fname_lc:
                    match = fname
                    break
            if match:
                break
        if match:
            found[tag] = os.path.join(similarity_dir, match)

    # If some expected tags are missing, also try heuristic: if a file contains 'resnet' but tag 'cnn' missing etc.
    if not found:
        print(f"No matching similarity files found in {similarity_dir}.")
    else:
        print("Found similarity files:")
        for k, v in found.items():
            print(f"  {k}: {v}")

    return found


def print_similarity_matrix(file_path):
    """Reads the similarity matrix from a CSV file and prints it."""
    try:
        #df = pd.read_csv(file_path)
        df = pd.read_csv(file_path, index_col=0)

        # Drop empty rows/cols
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

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
    print("File: " + file_path)


def print_important_env_vars():
    print("\nIMPORTANT ENV VARIABLES")
    print("-" * 50)

    for name, value in vars(env_vars).items():
        # skip internal stuff and large mapping dicts
        if name.startswith("__"):
            continue
        if name in ["SHORT_TO_LONG_NAME", "LONG_TO_SHORT_NAME"]:
            continue

        print(f"{name}: {value}")

    print("=" * 50)


def select_and_build_similarity_files(base_similarity_dir):
    chosen = select_similarity_folder(base_similarity_dir)
    if chosen is None:
        print("No similarity folder selected. Returning to menu.")
        return None

    similarity_files = build_similarity_files_from_dir(chosen)
    if not similarity_files:
        print("No similarity files found in the selected folder. Returning to menu.")
        return None

    return similarity_files


def read_files_from_url_or_folder(cross_results_dir):
    cross_results_files = []
    if "https" in cross_results_dir:
        response = requests.get(cross_results_dir)
        cross_results_files = [item for item in response.json() if item["type"] == "file"]
    else:
        if os.path.exists(cross_results_dir) == False:
            print("ERROR: Cross results directory does not exist:", cross_results_dir)
            return None
        # List files in local directory
        cross_results_files = []
        for fname in os.listdir(cross_results_dir):
            if fname.endswith(".txt"):
                cross_results_files.append({
                    "name": fname,
                    "download_url": os.path.join(cross_results_dir, fname)
                })
    return cross_results_files

# ---------- DATASET NAME NORMALIZATION ----------
def normalize_name(s):
    suffixes = ["_mean_pipeline", "_best_pipeline", ".txt"]
    for suffix in suffixes:
        s = s.replace(suffix, "")
    return s.strip()

def to_short_name(name):
    name = normalize_name(str(name)).strip()

    # already a short name
    if name in SHORT_TO_LONG_NAME:
        return name

    # long name -> short name
    if name in LONG_TO_SHORT_NAME:
        return LONG_TO_SHORT_NAME[name]

    # unknown name
    return None

def normalize_axis_labels(labels, axis_name="index"):
    normalized = []
    for label in labels:
        short = to_short_name(label)
        if short is None:
            print(f"[UNKNOWN {axis_name.upper()} LABEL] '{label}'")
            normalized.append(label)  # keep original for debugging
        else:
            normalized.append(short)
    return normalized
