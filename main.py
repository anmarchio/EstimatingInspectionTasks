import os

import env_vars
from env_vars import RESULTS_PATH, SIMILARITY_VALUES_FILE, \
    GITHUB_CROSS_APPLICATION_RESULTS_MEAN, GITHUB_CROSS_APPLICATION_RESULTS_BEST
from experiment_params_data import DATASETS
from src.models.resnet_embedding import compute_similarity_matrix
from src.plotting import plot_similarity_heatmap, show_similarity_results
from src.statistical_analysis import compute_correlation_analysis, compute_linear_regression, compute_mann_whitney_u, \
    bayesian_regression


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

def show_menu():
    print_important_env_vars()
    print("STUDY: Retrieval of Pipelines by Similarity")
    print("" + "=" * 50)
    print("[0] Exit")
    print("[1] Compute similarity between datasets")
    print("[2] Show similarity results")
    print("[3] Spearman Correlation")
    print("[4] Linear Regression")
    print("[5] Mann-Whitney-U Test")
    print("[6] Bayesian Linear Regression")
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
            print("[2] Show similarity results ...")
            result_dir = os.path.join(RESULTS_PATH)

            try:
                csv_files = [file for file in os.listdir(result_dir) if file.endswith(".csv")]
            except FileNotFoundError:
                print("No CSV files found in RESULTS_PATH.")
                continue

            if not csv_files:
                print("No CSV files found in RESULTS_PATH.")
                continue

            if len(csv_files) == 1:
                csv_file = csv_files[0]
                print(f"Reading similarity results from:\n `{os.path.join(result_dir, csv_file)}`")

                show_similarity_results(os.path.join(result_dir, csv_file))
                plot_similarity_heatmap(os.path.join(result_dir, csv_file))
                continue

            print("Found the following CSV files:")
            for idx, file in enumerate(csv_files):
                print(f"[{idx}] `{file}`")

            selection_idx = input("Select file index: ")
            try:
                selection_idx = int(selection_idx)
                if selection_idx < 0 or selection_idx >= len(csv_files):
                    print("Invalid selection.")
                    continue
            except ValueError:
                print("Invalid selection.")
                continue

            chosen_file = csv_files[selection_idx]
            chosen_path = os.path.join(result_dir, chosen_file)
            show_similarity_results(chosen_path)
            plot_similarity_heatmap(chosen_path)

        if selection == 3:
            # ------------------------------------------------
            # Correlation Analysis:
            # Compute Spearman correlation (rank-based, non-parametric, robust to non-linear relationships)
            # ------------------------------------------------
            print("[3] Performing Spearman Correlation Analysis ...")
            print("-> Computing Spearman correlation (rank-based, non-parametric, robust to non-linear relationships).")

            print("For MEAN MCC:")
            print("-" * 50)
            compute_correlation_analysis(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                         GITHUB_CROSS_APPLICATION_RESULTS_MEAN)

            print("For BEST MCC:")
            print("-" * 50)
            compute_correlation_analysis(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                         GITHUB_CROSS_APPLICATION_RESULTS_BEST)
        if selection == 4:
            # ------------------------------------------------
            # Linear Regression:
            # Fit a simple regression model to quantify how much similarity affects performance
            # ------------------------------------------------
            print("[4] Performing Linear Regression Analysis ...")
            print("-> Fitting a simple regression model to quantify how much similarity affects performance.")

            print("For MEAN MCC:")
            print("-" * 50)
            compute_linear_regression(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                      GITHUB_CROSS_APPLICATION_RESULTS_MEAN)

            print("For BEST MCC:")
            print("-" * 50)
            compute_linear_regression(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                      GITHUB_CROSS_APPLICATION_RESULTS_BEST)

        if selection == 5:
            # -------------------------------------------------
            # Mann-Whitney U:
            # Testing if high-similarity datasets lead to significantly better performance
            # -------------------------------------------------
            print("[5] Performing Mann-Whitney U Test ...")
            print("-> Testing if high-similarity datasets lead to significantly better performance.")

            print("For MEAN MCC:")
            print("-" * 50)
            compute_mann_whitney_u(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                   GITHUB_CROSS_APPLICATION_RESULTS_MEAN)

            print("For BEST MCC:")
            print("-" * 50)
            compute_mann_whitney_u(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                   GITHUB_CROSS_APPLICATION_RESULTS_BEST)

        if selection == 6:
            # -------------------------------------------------
            # Bayesian linear regression analysis:
            # Fit a Bayesian linear regression model to estimate the probability distribution of the effect of similarity on performance, providing uncertainty estimates.
            # -------------------------------------------------
            print("[6] Performing Bayesian Linear Regression Analysis ...")
            print("-> Fitting a Bayesian linear regression model to estimate the probability distribution of the effect "
                  "of similarity on performance, providing uncertainty estimates.")

            print("For MEAN MCC:")
            print("-" * 50)
            bayesian_regression(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                GITHUB_CROSS_APPLICATION_RESULTS_MEAN)

            print("For BEST MCC:")
            print("-" * 50)
            bayesian_regression(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                GITHUB_CROSS_APPLICATION_RESULTS_BEST)

        if selection > 6:
            print("Invalid selection. Please choose a valid option.")

    print("Exiting ....")


if __name__ == "__main__":
    main()
