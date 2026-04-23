import os

from env_vars import RESULTS_PATH, SIMILARITY_VALUES_FILE, \
    GITHUB_CROSS_APPLICATION_RESULTS_MEAN, GITHUB_CROSS_APPLICATION_RESULTS_BEST, SIMILARITY_DIR
from src.plotting import plot_similarity_heatmap, show_similarity_results
from src.similarity import compute_complexity_metrics, print_similarity_distribution
from src.statistical_analysis import compute_correlation_analysis, compute_linear_regression, compute_mann_whitney_u, \
    bayesian_regression, linear_regression_on_multiple_similarity_metrics, perform_pipeline_reuse_multimetric_analysis
from src.utils import print_important_env_vars, select_dir, select_and_build_similarity_files


def show_help():
    print("\nHELP: Overview of available commands")
    print("-" * 50)
    print("h : Show help")
    print("0 : Exit")
    print("1 : Compute similarities")
    print("2 : Display similarity plots")
    print("3 : Spearman correlation")
    print("4 : Linear regression (CNN similarity)")
    print("5 : Mann-Whitney U test")
    print("6 : Bayesian regression")
    print("7 : Regression with multiple similarity metrics")
    print("8 : Pipeline reuse & multi-metric analysis")
    print("9 : Full analysis (3 & 7 & 8)")
    print("-" * 50)
    input("Press Enter to return to the menu...")


def show_menu():
    print_important_env_vars()

    # ANSI colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print("STUDY: Retrieval of Pipelines by Similarity")
    print("=" * 50)

    # Help + Exit
    print("[h] Help")
    print(f"{RED}[0] EXIT{RESET}")

    # Preparation Section
    print(f"{BLUE}{BOLD}PREPARATION{RESET}")
    print("[1] Compute similarity")
    print("[2] Show similarity plots")

    # Single Insights Section
    print(f"{GREEN}{BOLD}SINGLE INSIGHTS{RESET}")
    print("[3] Spearman Correlation")
    print("[4] Linear Regression with CNN Similarity")
    print("[5] Mann-Whitney-U Test (less relevant)")
    print("[6] Bayesian Linear Regression (not analyzed)")
    print("[7] Linear Regression on Multiple Similarity Metrics")
    print("[8] Pipeline Reuse & Multi-metric Analysis")

    # Full Analysis Section
    print(f"{YELLOW}{BOLD}FULL ANALYSIS{RESET}")
    print("[9] Run [3] & [7] & [8] For full analysis")

    print("-" * 50)

    selection = input("Go to: ")

    # Handle help separately
    if selection.lower() == 'h':
        show_help()  # <-- make sure this function exists
        return -1

    try:
        selection = int(selection)
    except ValueError:
        print("Invalid selection. Please enter a number or 'h'.")
        return -1

    return selection


def print_capital_separator(label):
    print("\n" + "=" * 80)
    print("\033[1;33m" + f"{'  >>> FOR ' + label + ' <<<  ':^80}" + "\033[0m")
    print("=" * 80)


def main():
    running = True
    while running:
        selection = show_menu()

        if selection == 0:
            running = False

        if selection == 1:
            print("[1] Computing similarity metrics between datasets pairs...")
            compute_complexity_metrics()

        if selection == 2:
            print("[2] Show similarity results ...")

            input("Full matrix view? (y/n): ")
            full = (input().lower() == 'y')
            results_paths = select_dir(SIMILARITY_DIR)

            if results_paths is None or results_paths == []:
                continue

            for result_path in results_paths:
                if full:
                    show_similarity_results(result_path)
                print_similarity_distribution(result_path)
                plot_similarity_heatmap(result_path)

        if selection == 3:
            # ------------------------------------------------
            # Correlation Analysis:
            # Compute Spearman correlation (rank-based, non-parametric, robust to non-linear relationships)
            # ------------------------------------------------
            print("[3] Performing Correlation Analysis ...")
            print("-> Computing correlation (rank-based, non-parametric, robust to non-linear relationships).")

            similarity_files = select_and_build_similarity_files(SIMILARITY_DIR)

            if not similarity_files:
                continue

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print_capital_separator(label)
                for key in similarity_files.keys():
                    print(f">>> START analysis for similarity file: {similarity_files[key]}")
                    compute_correlation_analysis(similarity_files[key],
                                                 target)
                    print(f"<<< END analysis for similarity file: {similarity_files[key]}")

        if selection == 4:
            # ------------------------------------------------
            # Linear Regression on CNN Similarity:
            # Fit a simple regression model to quantify how much CNN similarity affects performance
            # ------------------------------------------------
            print("[4] Linear Regression on CNN Similarity ...")
            print("-> Fitting a simple regression model to quantify how much similarity affects performance.")

            similarity_files = select_and_build_similarity_files(SIMILARITY_DIR)

            if not similarity_files:
                continue

            for file in similarity_files:
                print("Only use CNN embeddings ...")
                if "cnn" in file or "resnet" in file or "embedding" in file:
                    continue
                else:
                    similarity_files.remove(file)

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print_capital_separator(label)
                for key in similarity_files.keys():
                    print(f"Using similarity file: {similarity_files[key]}")
                    compute_linear_regression(similarity_files[key], target)

        if selection == 5:
            # -------------------------------------------------
            # Mann-Whitney U:
            # Testing if high-similarity datasets lead to significantly better performance
            # -------------------------------------------------
            print("[5] Performing Mann-Whitney U Test ...")
            print("-> Testing if high-similarity datasets lead to significantly better performance.")

            similarity_files = select_and_build_similarity_files(SIMILARITY_DIR)

            if not similarity_files:
                continue

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print_capital_separator(label)
                for key in similarity_files.keys():
                    print(f"Using similarity file: {similarity_files[key]}")
                    compute_mann_whitney_u(similarity_files[key], target)

        if selection == 6:
            # -------------------------------------------------
            # Bayesian linear regression analysis:
            # Fit a Bayesian linear regression model to estimate the probability distribution of the effect of similarity on performance, providing uncertainty estimates.
            # -------------------------------------------------
            print("[6] Performing Bayesian Linear Regression Analysis ...")
            print(
                "-> Fitting a Bayesian linear regression model to estimate the probability distribution of the effect "
                "of similarity on performance, providing uncertainty estimates.")

            similarity_files = select_and_build_similarity_files(SIMILARITY_DIR)

            if not similarity_files:
                continue

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print_capital_separator(label)
                for key in similarity_files.keys():
                    print(f"Using similarity file: {similarity_files[key]}")
                    bayesian_regression(similarity_files[key], target)

        if selection == 7:
            # Insight analysis:
            # compute per-pipeline metrics (mean, median, std, min, transfer_rate (>0),
            # strong_transfer_rate (>0.1)).
            # Filter pipelines with transfer_rate > 0.5
            # and rank by mean_cross_score (or by a combined score).
            print("[7] Linear Regression on Multiple Similarity Metrics ...")
            print("-> Fitting a linear regression model with multiple metrics (mean/median cross_score, "
                  "transfer_rate) and statistical metrics.")

            similarity_files = select_and_build_similarity_files(SIMILARITY_DIR)

            if not similarity_files:
                continue

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print_capital_separator(label)
                for key in similarity_files.keys():
                    print(f"Using similarity file: {similarity_files[key]}")
                    linear_regression_on_multiple_similarity_metrics(similarity_files[key],
                                                                     target)

        if selection == 8:
            print("[8] Pipeline Resuse & Multi-metric Analysis ...")
            print("-> Performing pipeline resuse and multi-metric analysis.")

            similarity_files = select_and_build_similarity_files(SIMILARITY_DIR)

            if not similarity_files:
                continue

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print_capital_separator(label)
                perform_pipeline_reuse_multimetric_analysis(similarity_files, target)

        if selection == 9:
            print("[9] Running full analysis (3 & 7 & 8) ...")
            print("-> Running the full analysis pipeline (3 & 7 & 8) for comprehensive insights.")

            similarity_files = select_and_build_similarity_files(SIMILARITY_DIR)

            if not similarity_files:
                continue

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print_capital_separator(label)
                for key in similarity_files.keys():
                    print("--- CORRELATION ANALYSIS ---")
                    compute_correlation_analysis(similarity_files[key],
                                                 target)

                    print("--- MULTIPLE METRICS LINEAR REGRESSION ---")
                    linear_regression_on_multiple_similarity_metrics(similarity_files[key],
                                                                     target)
                print("--- PIPELINE REUSE & MULTI-METRIC ANALYSIS ---")
                perform_pipeline_reuse_multimetric_analysis(similarity_files, target)

        if selection > 9:
            print("Invalid selection. Please choose a valid option.")

    print("Exiting ....")


if __name__ == "__main__":
    main()
