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
            similarity_df = pd.read_csv(f, index_col=0)
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
        with open(os.path.join(result_dir, file), 'r', encoding='utf-8') as f:
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
                        value = float(cell)
                        values.append(f"{value:.2f}")
                    latex_line = " & ".join(values) + " \\\\"
                    print(latex_line)

                idx += 1
            print("\hline")
            print("\end{tabular}%")
            print("}")
            print("\end{table}")