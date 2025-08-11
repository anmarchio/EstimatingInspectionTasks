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
        with open(file_path, 'r', encoding='utf-8') as f:
            similarity_df = pd.read_csv(file_path, index_col=0).dropna(axis=1, how="all")
        print("Similarity matrix loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    if similarity_df is not None:
        # --- Replace labels with numbers ---
        n = len(similarity_df)
        num_labels = list(range(1, n + 1))
        similarity_df.index = num_labels
        similarity_df.columns = num_labels

        # --- Plot heatmap without annotations ---
        sns.heatmap(similarity_df, annot=False, cmap="coolwarm", square=True,
                    cbar_kws={"label": "Cosine Similarity"})
        # plt.title("Cosine Similarity of Datasets")
        plt.tight_layout()
        plt.savefig(file_path.replace('.csv', '_heatmap.png'))
        plt.show()
    else:
        raise ValueError("No similarity data to plot.")


def show_similarity_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            if idx == 0:
                cells = line.strip().split(',')
                print("\\begin{table}[h]")
                print("\centering")
                print("\caption{Dataset ID and Descriptions}")
                print("\label{tab:dataset_description}")
                print("\\resizebox{0.5\columnwidth}{!}{%")
                print("\\begin{tabular}{cc}")
                print("\hline")
                print("ID & Dataset Name \\\\")
                print("\hline")
                cdx = 1
                header = ""
                for c in cells:
                    if c != '':
                        c = c.replace('_', '\\_')
                        print(f"{cdx} & {c} \\\\")
                        if cdx == 1:
                            header = f"{cdx} "
                        else:
                            header += f"& {cdx} "
                    cdx += 1
                header += " \\\\"
                print("\hline")
                print("\end{tabular}%")
                print("}")
                print("\end{table}")

                print("\\begin{table*}[t]")
                print("\centering")
                print("\caption{Cosine Similarity between all datasets}")
                print("\label{tab:similarity-matrix}")
                print("\\resizebox{2.0\columnwidth}{!}{%")
                header_format = "c" * len(cells)
                print("\\begin{tabular}{" + header_format + "}")
                print("\hline")
                print(header)
                print("\hline")
            else:
                cells = line.strip().split(',')
                values = []
                for cell in cells:
                    try:
                        value = float(cell)
                        values.append(f"{value:.2f}")
                    except:
                        continue
                latex_line = " & ".join(values) + " \\\\"
                print(latex_line)

            idx += 1
        print("\hline")
        print("\end{tabular}%")
        print("}")
        print("\end{table}")