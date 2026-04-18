import os

import env_vars
from env_vars import RESULTS_PATH, SIMILARITY_VALUES_FILE, \
    GITHUB_CROSS_APPLICATION_RESULTS_MEAN, GITHUB_CROSS_APPLICATION_RESULTS_BEST, WDIR
from src.plotting import plot_similarity_heatmap, show_similarity_results
from src.similarity import select_complexity_function
from src.statistical_analysis import compute_correlation_analysis, compute_linear_regression, compute_mann_whitney_u, \
    bayesian_regression, linear_regression_on_multiple_similarity_metrics, perform_pipeline_reuse_multimetric_analysis


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
    print("[0] EXIT")
    print("[1] Compute similarity between datasets")
    print("[2] Show similarity results")
    print("[3] Spearman Correlation")
    print("[4] Linear Regression with CNN Similarity")
    print("[5] Mann-Whitney-U Test (less relevant)")
    print("[6] Bayesian Linear Regression (not analyzed)")
    print("[7] Linear Regresssion on Multiple Similarity Metrics")
    print("[8] Pipeline Resuse & Multi-metric Analysis")
    print("" + "-" * 50)
    selection = input("Go to: ")

    try:
        selection = int(selection)
    except ValueError:
        print("Invalid selection. Please enter a number.")
        return -1

    return selection


def select_file(results_dir):
    try:
        csv_files = [file for file in os.listdir(results_dir) if file.endswith(".csv")]
    except FileNotFoundError:
        print(f"No CSV files found in {results_dir}.")
        return None

    if not csv_files:
        print(f"No CSV files found in {results_dir}.")
        return None

    if len(csv_files) == 1:
        csv_file = csv_files[0]
        print(f"Reading similarity results from:\n `{os.path.join(results_dir, csv_file)}`")

        show_similarity_results(os.path.join(results_dir, csv_file))
        plot_similarity_heatmap(os.path.join(results_dir, csv_file))
        return None

    print("Found the following CSV files:")
    for idx, file in enumerate(csv_files):
        print(f"[{idx}] `{file}`")

    selection_idx = input("Select file index: ")
    try:
        selection_idx = int(selection_idx)
        if selection_idx < 0 or selection_idx >= len(csv_files):
            print("Invalid selection.")
            return None
    except ValueError:
        print("Invalid selection.")
        return None

    chosen_file = csv_files[selection_idx]
    return os.path.join(RESULTS_PATH, chosen_file)


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


def main():
    running = True
    while running:
        selection = show_menu()

        if selection == 0:
            running = False

        if selection == 1:
            print("[1] Computing similarity between datasets pairs...")
            results_paths = compute_complexity_metrics()

            for result_path in results_paths:
                plot_similarity_heatmap(result_path)

        if selection == 2:
            print("[2] Show similarity results ...")

            chosen_path = select_file(RESULTS_PATH)

            if chosen_path is None:
                continue

            show_similarity_results(chosen_path)
            plot_similarity_heatmap(chosen_path)

        if selection == 3:
            # ------------------------------------------------
            # Correlation Analysis:
            # Compute Spearman correlation (rank-based, non-parametric, robust to non-linear relationships)
            # ------------------------------------------------
            print("[3] Performing Spearman Correlation Analysis ...")
            print("-> Computing Spearman correlation (rank-based, non-parametric, robust to non-linear relationships).")

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print(f"For {label}:")
                print("-" * 50)
                compute_correlation_analysis(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                         target)

        if selection == 4:
                # ------------------------------------------------
                # Linear Regression on CNN Similarity:
                # Fit a simple regression model to quantify how much CNN similarity affects performance
                # ------------------------------------------------
                print("[4] Linear Regression on CNN Similarity ...")
                print("-> Fitting a simple regression model to quantify how much similarity affects performance.")

                for label, target in [
                    ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                    ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
                ]:
                    print(f"For {label}:")
                    print("-" * 50)
                    compute_linear_regression(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE), target)

        if selection == 5:
            # -------------------------------------------------
            # Mann-Whitney U:
            # Testing if high-similarity datasets lead to significantly better performance
            # -------------------------------------------------
            print("[5] Performing Mann-Whitney U Test ...")
            print("-> Testing if high-similarity datasets lead to significantly better performance.")

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print(f"For {label}:")
                print("-" * 50)
                compute_mann_whitney_u(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE), target)

        if selection == 6:
            # -------------------------------------------------
            # Bayesian linear regression analysis:
            # Fit a Bayesian linear regression model to estimate the probability distribution of the effect of similarity on performance, providing uncertainty estimates.
            # -------------------------------------------------
            print("[6] Performing Bayesian Linear Regression Analysis ...")
            print("-> Fitting a Bayesian linear regression model to estimate the probability distribution of the effect "
                  "of similarity on performance, providing uncertainty estimates.")

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print(f"For {label}:")
                print("-" * 50)
                bayesian_regression(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                    target)

        if selection == 7:
            # Insight analysis:
            # compute per-pipeline metrics (mean, median, std, min, transfer_rate (>0),
            # strong_transfer_rate (>0.1)).
            # Filter pipelines with transfer_rate > 0.5
            # and rank by mean_cross_score (or by a combined score).
            print("[7] Linear Regresssion on Multiple Similarity Metrics ...")
            print("-> Fitting a linear regression model with multiple metrics (mean/median cross_score, "
                  "transfer_rate) and statisticial metrics.")

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print(f"For {label}:")
                print("-" * 50)
                linear_regression_on_multiple_similarity_metrics(os.path.join(RESULTS_PATH, SIMILARITY_VALUES_FILE),
                                                                 target)

        if selection == 8:
            print("[8] Pipeline Resuse & Multi-metric Analysis ...")
            print("-> Performing pipeline resuse and multi-metric analysis.")


            # Let user pick which timestamped similarity folder to use
            base_similarity_dir = os.path.join(WDIR, "results", "similarity")
            chosen_folder = select_similarity_folder(base_similarity_dir)
            if chosen_folder is None:
                print("No similarity folder selected. Returning to menu.")
                continue

            similarity_files = build_similarity_files_from_dir(chosen_folder)

            if not similarity_files:
                print("No similarity files found in the selected folder. Returning to menu.")
                continue

            for label, target in [
                ("MEAN MCC", GITHUB_CROSS_APPLICATION_RESULTS_MEAN),
                ("BEST MCC", GITHUB_CROSS_APPLICATION_RESULTS_BEST),
            ]:
                print(f"For {label}:")
                print("-" * 50)
                perform_pipeline_reuse_multimetric_analysis(similarity_files, target)

        if selection > 8:
            print("Invalid selection. Please choose a valid option.")

    print("Exiting ....")


if __name__ == "__main__":
    main()
