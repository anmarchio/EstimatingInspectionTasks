import os

import pandas as pd

import seaborn as sns
from glob import glob
from matplotlib import pyplot as plt

from env_vars import RESULTS_PATH
from src.data_handling import print_table, read_df


def plot_similarity_heatmap(file_path):
    """Reads the similarity matrix from a CSV file."""
    similarity_df = None
    try:
        similarity_df = pd.read_csv(file_path, index_col=0)
        print("Similarity matrix loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    if similarity_df is not None:
        sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Similarity Heatmap")
        plt.savefig(file_path.replace('.csv', '_heatmap.png'))
        plt.show()
    else:
        raise ValueError("No similarity data to plot.")


def show_similarity_results():
    result_dir = os.path.join(RESULTS_PATH)

    csv_files = [file for file in os.listdir(result_dir) if file.endswith(".csv")]

    for file in csv_files:
        similarity_df = read_df(file)
        print(similarity_df.to_string())
