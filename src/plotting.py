import os

import pandas as pd

import seaborn as sns
from glob import glob
from matplotlib import pyplot as plt

from env_vars import RESULTS_PATH
from src.data_handling import print_table, read_df


def plot_similarity_heatmap(file_path):
    df = pd.read_csv(file_path, index_col=0)

    # Drop empty rows/cols
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Normalize labels so they can match
    df.index = df.index.astype(str).str.strip()
    df.columns = pd.Index(df.columns).astype(str).str.strip()

    # Debug: how many overlap?
    common = df.index.intersection(df.columns)
    print("shape after drop:", df.shape)
    print("common count:", len(common))

    if len(common) == 0:
        # show a few unmatched examples
        print("first 10 rows:", df.index[:10].tolist())
        print("first 10 cols:", list(df.columns[:10]))
        raise ValueError("No overlapping labels between index and columns. "
                         "Your CSV row names and column names don't match.")

    df = df.loc[common, common]
    print("shape after align:", df.shape)

    # Relabel 1..n
    n = df.shape[0]
    labels = list(range(1, n + 1))
    df.index = labels
    df.columns = labels

    sns.heatmap(df, annot=False, cmap="coolwarm", square=True,
                cbar_kws={"label": "Cosine Similarity"})
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(file_path.replace(".csv", "_heatmap.png"))
    plt.show()


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