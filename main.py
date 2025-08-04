import os

from env_vars import MANUAL_HDEV_DIR, RESULTS_PATH
from experiment_params_data import DATASETS
from src.models.resnet_embedding import compute_similarity_matrix
from src.plotting import plot_similarity_heatmap, show_similarity_results
from src.statistical_analysis import compute_correlation_analysis, compute_linear_regression, compute_mann_whitney_u


def show_menu():
    print("STUDY: Retrieval of Pipelines by Similarity")
    print("" + "=" * 50)
    print("[0] Exit")
    print("[1] Compute similarity between datasets")
    print("[2] Show similarity results")
    print("[3] Spearman Correlation")
    print("[4] Linear Regression")
    print("[5] Mann-Whitney-U Test")
    print("" + "-" * 50)
    selection = input("Go to: ")

    try:
        selection = int(selection)
    except ValueError:
        print("Invalid selection. Please enter a number.")
        return -1

    return selection


def main():
    running = True
    while running:
        selection = show_menu()

        if selection == 0:
            running = False

        if selection == 1:
            # ------------------------------------------------
            # Compute similarity between dataset pairs
            print("[1] Computing similarity between datasets...")
            # ------------------------------------------------
            result_path = compute_similarity_matrix(list(DATASETS.keys()), [v['train'] for k, v in DATASETS.items()])

            plot_similarity_heatmap(result_path)

        if selection == 2:
            print("[2] Showing similarity results ...")
            show_similarity_results()

        if selection == 3:
            # ------------------------------------------------
            # Correlation Analysis:
            # Compute Spearman correlation (rank-based, non-parametric, robust to non-linear relationships)
            # ------------------------------------------------
            compute_correlation_analysis(os.path.join(RESULTS_PATH, "results-20250803.csv"), MANUAL_HDEV_DIR)

        if selection == 4:
            # ------------------------------------------------
            # Linear Regression:
            # Fit a simple regression model to quantify how much similarity influences performance
            # ------------------------------------------------
            compute_linear_regression()

        if selection == 5:
            # -------------------------------------------------
            # Mann-Whitney U:
            # Does high-similarity lead to significantly better performance
            # -------------------------------------------------
            compute_mann_whitney_u()

        if selection > 4:
            print("Invalid selection. Please choose a valid option.")

    print("Exiting ....")


if __name__ == "__main__":
    main()
