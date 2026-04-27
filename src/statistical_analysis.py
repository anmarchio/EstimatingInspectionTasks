import os.path
import warnings

import arviz as az
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import requests
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr, pearsonr
from skimage.feature import graycomatrix, graycoprops
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest

from env_vars import LONG_TO_SHORT_NAME
from src.utils import to_short_name, normalize_axis_labels, normalize_name, read_files_from_url_or_folder


def extract_image_statistics(image):
    """Extracts statistical features from a grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Intensity features
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # Edge density (Sobel filter)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge_density = np.mean(edges > 50)

    # Texture features using GLCM (Gray-Level Co-Occurrence Matrix)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    return [mean_intensity, std_intensity, edge_density, contrast, entropy]


def load_and_prepare_similarity_and_cross_results(similarity_filepath, cross_results_dir):
    """
    This function loads the similarity matrix and cross-application results, and prepares a combined dataframe for analysis. It includes:
    - Loading the similarity matrix from a CSV file and ensuring it is properly indexed
    - Fetching the list of cross-application result files from the GitHub repository
    - For each result file, extracting the source and target dataset names, and matching them to
        the similarity matrix to retrieve the corresponding similarity score
    - Compiling all matched pairs into a single dataframe with columns for source dataset, target dataset, similarity score, and cross-application performance score
    """
    # --- Step 1: Load similarity matrix
    similarity_df = None
    with open(similarity_filepath, 'r', encoding='utf-8') as f:
        similarity_df = pd.read_csv(similarity_filepath, index_col=0,
                                    usecols=lambda col: col != similarity_filepath.split('/')[-1])

    if similarity_df is None:
        raise ValueError("No similarity data found!")

    # --- Step 2: Prepare to load cross-application results
    all_rows = []

    # Normalize column and index names for matching
    similarity_df.columns = similarity_df.columns.astype(str).str.strip()
    similarity_df.index = similarity_df.index.astype(str).str.strip()

    # Safely Normalize Names to short form
    similarity_df.index = normalize_axis_labels(similarity_df.index, "index")
    similarity_df.columns = normalize_axis_labels(similarity_df.columns, "column")

    # --- Step 3: Load each result file and collect scores
    cross_results_files = read_files_from_url_or_folder(cross_results_dir)

    lookup_errors = ""

    for file in cross_results_files:
        # src_dataset = normalize_name(fname)
        # path = os.path.join(cross_results_dir, fname)

        raw_url = file["download_url"]  # direct raw file link
        src_dataset = file["name"]

        if not src_dataset.endswith("_pipeline.txt"):
            continue

        # Download file content directly into pandas
        df = pd.read_csv(raw_url, sep=';', engine='python')

        for _, row in df.iterrows():
            if row.iloc[2] is None:
                warnings.warn(
                    f"Skipping empty row: {row.tolist()} in file: {src_dataset}",
                    UserWarning
                )
                continue
            elif (row.iloc[0] == 'Pipeline'
                    and row.iloc[1] == ' OriginalScore'
                    and row.iloc[2] == ' CrossApplication'
                    and row.iloc[3] == ' CrossScore'
                  ):
                print(f"Skipping header row: {row.tolist()} in file: {src_dataset}")
                continue
            else:
                tgt_dataset = normalize_name(row.iloc[2])
                src_dataset = normalize_name(src_dataset)

            # ----- Check Normalization Error -----
            src_key = to_short_name(src_dataset)
            tgt_key = to_short_name(tgt_dataset)

            if src_key is None or tgt_key is None:
                lookup_errors += (
                    f"[NAME NORMALIZATION ERROR] "
                    f"src='{src_dataset}' -> {src_key}, "
                    f"tgt='{tgt_dataset}' -> {tgt_key}\n"
                )
                #continue

            if src_key not in similarity_df.index or tgt_key not in similarity_df.columns:
                lookup_errors += (
                    f"[MATRIX LOOKUP ERROR] "
                    f"src_key='{src_key}' in index={src_key in similarity_df.index}, "
                    f"tgt_key='{tgt_key}' in columns={tgt_key in similarity_df.columns} | "
                    f"original src='{src_dataset}', tgt='{tgt_dataset}'\n"
                )
                #continue
            # ----------------------------------

            try:
                #similarity = similarity_df.loc[
                #    LONG_TO_SHORT_NAME[src_key],
                #    LONG_TO_SHORT_NAME[tgt_key]
                #]
                similarity = similarity_df.loc[src_key, tgt_key]
            except Exception as e:
                #print(f"Skipping pair due to error: {e} in file: {src_dataset} with target: {tgt_dataset}")
                continue

            all_rows.append({
                "source": src_dataset,
                "target": tgt_dataset,
                "similarity": similarity,
                "cross_score": float(row.iloc[3])
            })

    print(lookup_errors)

    # --- Step 4: Create dataframe for analysis
    correlation_df = pd.DataFrame(all_rows)

    return similarity_df, correlation_df


def compute_linear_regression(similarity_filepath, cross_results_dir):
    """
    This function performs a linear regression analysis to examine the relationship between dataset similarity and cross-application performance. It includes:
    - Loading and preparing the similarity matrix and cross-application results
    - Fitting an OLS regression model with similarity as the predictor and cross-application performance as the target variable
    - Interpreting the regression results in terms of model fit, coefficient significance, and practical implications
    """
    similarity_df, correlation_df = load_and_prepare_similarity_and_cross_results(similarity_filepath,
                                                                                  cross_results_dir)

    # --- Step 6: Compute linear regression for the entire dataset
    X = correlation_df['similarity']
    y = correlation_df['cross_score']
    X = sm.add_constant(X)  # adds intercept
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # --- Step 7: Interpret results ---
    interpretation = interpret_ols(
        model,
        predictor_name="similarity",
        target_name="cross_score"
    )

    print("\n--- Interpretation of Linear Regression Results ---")
    print(interpretation)


def interpret_ols(results, predictor_name="similarity", target_name="cross_score"):
    """
    This function takes the results of an OLS regression and generates a detailed interpretation of the findings, including:
    - Overall model fit (R², adjusted R², F-statistic)
    - Coefficient interpretation for the predictor variable (magnitude, direction, significance)
    - Residual diagnostics (Durbin-Watson, Jarque-Bera, Omnibus tests)
    - A concise conclusion summarizing the practical implications of the results.
    """
    statements = []

    # ---------- overall model ----------
    r2 = results.rsquared
    adj_r2 = results.rsquared_adj
    f_stat = results.fvalue
    f_p = results.f_pvalue
    n = int(results.nobs)

    if r2 < 0.02:
        r2_text = "very low"
    elif r2 < 0.13:
        r2_text = "low"
    elif r2 < 0.26:
        r2_text = "moderate"
    else:
        r2_text = "high"

    statements.append(
        f"The linear regression model examined whether {predictor_name} predicts {target_name} "
        f"based on {n} observations."
    )

    statements.append(
        f"The model shows {r2_text} explanatory power (R² = {r2:.3f}, adjusted R² = {adj_r2:.3f}), "
        f"meaning that {predictor_name} explains about {r2*100:.1f}% of the variance in {target_name}."
    )

    if f_p < 0.001:
        statements.append(
            f"The overall regression model is statistically significant "
            f"(F = {f_stat:.2f}, p < 0.001)."
        )
    elif f_p < 0.05:
        statements.append(
            f"The overall regression model is statistically significant "
            f"(F = {f_stat:.2f}, p = {f_p:.3g})."
        )
    else:
        statements.append(
            f"The overall regression model is not statistically significant "
            f"(F = {f_stat:.2f}, p = {f_p:.3g})."
        )

    # ---------- predictor ----------
    if predictor_name in results.params.index:
        coef = results.params[predictor_name]
        std_err = results.bse[predictor_name]
        t_val = results.tvalues[predictor_name]
        p_val = results.pvalues[predictor_name]
        ci_low, ci_high = results.conf_int().loc[predictor_name]

        direction = "positive" if coef > 0 else "negative"

        abs_coef = abs(coef)
        if abs_coef < 0.1:
            magnitude = "very small"
        elif abs_coef < 0.3:
            magnitude = "small"
        elif abs_coef < 0.5:
            magnitude = "moderate"
        else:
            magnitude = "large"

        if p_val < 0.001:
            sig_text = "statistically significant (p < 0.001)"
        elif p_val < 0.05:
            sig_text = f"statistically significant (p = {p_val:.3g})"
        else:
            sig_text = f"not statistically significant (p = {p_val:.3g})"

        statements.append(
            f"The coefficient for {predictor_name} is {coef:.4f} (SE = {std_err:.4f}, "
            f"t = {t_val:.2f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]), indicating a "
            f"{direction} and {magnitude} relationship with {target_name}. "
            f"This effect is {sig_text}."
        )

        statements.append(
            f"In practical terms, a one-unit increase in {predictor_name} is associated with "
            f"an average change of {coef:.4f} units in {target_name}."
        )

    # ---------- intercept ----------
    if "const" in results.params.index:
        intercept = results.params["const"]
        intercept_p = results.pvalues["const"]
        statements.append(
            f"The intercept is {intercept:.4f} (p = {intercept_p:.3g}), representing the expected "
            f"value of {target_name} when {predictor_name} equals zero."
        )

    # ---------- residual diagnostics ----------
    residuals = results.resid

    dw = durbin_watson(residuals)
    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
    omni_stat, omni_p = omni_normtest(residuals)

    if 1.5 <= dw <= 2.5:
        statements.append(
            f"The Durbin-Watson statistic is {dw:.3f}, suggesting no strong autocorrelation in the residuals."
        )
    elif dw < 1.5:
        statements.append(
            f"The Durbin-Watson statistic is {dw:.3f}, suggesting possible positive autocorrelation in the residuals."
        )
    else:
        statements.append(
            f"The Durbin-Watson statistic is {dw:.3f}, suggesting possible negative autocorrelation in the residuals."
        )

    if jb_p < 0.05:
        statements.append(
            f"The Jarque-Bera test indicates that the residuals deviate from normality "
            f"(JB = {jb_stat:.2f}, p < 0.05, skew = {skew:.3f}, kurtosis = {kurtosis:.3f})."
        )
    else:
        statements.append(
            f"The Jarque-Bera test does not indicate a significant deviation from normality "
            f"(JB = {jb_stat:.2f}, p = {jb_p:.3g})."
        )

    if omni_p < 0.05:
        statements.append(
            f"The Omnibus test also suggests that the residuals are not normally distributed "
            f"(Omnibus = {omni_stat:.2f}, p < 0.05)."
        )
    else:
        statements.append(
            f"The Omnibus test does not indicate a significant deviation from normality "
            f"(Omnibus = {omni_stat:.2f}, p = {omni_p:.3g})."
        )

    # ---------- concise conclusion ----------
    if predictor_name in results.params.index:
        coef = results.params[predictor_name]
        p_val = results.pvalues[predictor_name]

        if p_val < 0.05 and r2 < 0.02:
            statements.append(
                f"Overall, {predictor_name} is a statistically significant predictor of {target_name}, "
                f"but its practical explanatory contribution is very limited."
            )
        elif p_val < 0.05:
            statements.append(
                f"Overall, {predictor_name} is a statistically significant predictor of {target_name}."
            )
        else:
            statements.append(
                f"Overall, {predictor_name} does not appear to be a reliable predictor of {target_name}."
            )

    statements.append("...\n")

    return "\n".join(statements)

def safe_pearson(x, y, label=""):
    # drop NaNs pairwise
    df = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(df) < 2:
        print(f"[SKIP] Not enough data for Pearson ({label}): n={len(df)}")
        return np.nan, np.nan

    return pearsonr(df["x"], df["y"])

def compute_correlation_analysis(similarity_filepath, cross_results_dir):
    """
    This function performs a correlation analysis to examine the relationship between dataset similarity and cross-application performance. It includes:
    - Loading and preparing the similarity matrix and cross-application results
    - Computing Pearson and Spearman correlation
    - Interpreting the correlation results in terms of strength, direction, and statistical significance
    - Analyzing correlations within different similarity ranges (low, medium, high)
    - Analyzing correlations within different bins of similarity (e.g., 0-0.2, 0.2-0.4, etc.)
    """
    _, correlation_df = load_and_prepare_similarity_and_cross_results(similarity_filepath, cross_results_dir)

    try:
        # --- Step 5: Compute correlations
        pearson_corr, pearson_p = pearsonr(correlation_df["similarity"], correlation_df["cross_score"])
        spearman_corr, spearman_p = spearmanr(correlation_df["similarity"], correlation_df["cross_score"])

        print("✅ Pearson correlation:", pearson_corr, " (p =", pearson_p, ")")
        print("✅ Spearman correlation:", spearman_corr, " (p =", spearman_p, ")")

        # --- Step 6: Compute correlations for different similarity ranges
        print("\n--- Analysis for Bin Size: [-1.0;0.5[, [0.5;0.7[, [0.7;1.0]  ---")
        low_similarity = correlation_df[correlation_df["similarity"] < 0.5]
        medium_similarity = correlation_df[(correlation_df["similarity"] >= 0.5) & (correlation_df["similarity"] < 0.7)]
        high_similarity = correlation_df[correlation_df["similarity"] >= 0.7]

        # Low similarity
        low_pearson_corr, low_pearson_p = safe_pearson(
            low_similarity["similarity"],
            low_similarity["cross_score"],
            label="low similarity"
        )
        low_spearman_corr, low_spearman_p = spearmanr(low_similarity["similarity"], low_similarity["cross_score"])
        print("✅ Low Similarity (s < 0.5)- Pearson correlation:", low_pearson_corr, " (p =", low_pearson_p, ")")
        print("✅ Low Similarity (s < 0.5) - Spearman correlation:", low_spearman_corr, " (p =", low_spearman_p, ")")

        # Medium similarity
        medium_pearson_corr, medium_pearson_p = safe_pearson(
            medium_similarity["similarity"],
            medium_similarity["cross_score"],
            label="low similarity"
        )
        medium_spearman_corr, medium_spearman_p = spearmanr(medium_similarity["similarity"],
                                                            medium_similarity["cross_score"])
        print("✅ Medium Similarity (0.5 <= s < 0.7) - Pearson correlation:", medium_pearson_corr, " (p =", medium_pearson_p, ")")
        print("✅ Medium Similarity (0.5 <= s < 0.7) - Spearman correlation:", medium_spearman_corr, " (p =", medium_spearman_p, ")")

        # High similarity
        high_pearson_corr, high_pearson_p = safe_pearson(
            high_similarity["similarity"],
            high_similarity["cross_score"],
            label="low similarity"
        )
        high_spearman_corr, high_spearman_p = spearmanr(high_similarity["similarity"], high_similarity["cross_score"])
        print("✅ High Similarity (0.7 <= s <= 1.0) - Pearson correlation:", high_pearson_corr, " (p =", high_pearson_p, ")")
        print("✅ High Similarity (0.7 <= s <= 1.0) - Spearman correlation:", high_spearman_corr, " (p =", high_spearman_p, ")")

        # ------------ Analyze Bins of different sizes ----------
        bin_sizes = [0.2, 0.1]
        for bin_size in bin_sizes:
            print(f"\n--- Analysis for Bin Size: {bin_size} ---")
            bins = np.arange(0, 1.1, bin_size)
            correlation_df['bin'] = pd.cut(correlation_df['similarity'], bins, include_lowest=True)

            for bin_range, group in correlation_df.groupby('bin'):
                if len(group) > 1:  # Ensure there are enough data points
                    bin_pearson_corr, bin_pearson_p = pearsonr(group["similarity"], group["cross_score"])
                    bin_spearman_corr, bin_spearman_p = spearmanr(group["similarity"], group["cross_score"])
                    print(f"Bin {bin_range}: Pearson correlation = {bin_pearson_corr:.3f} (p = {bin_pearson_p:.5f}), "
                          f"Spearman correlation = {bin_spearman_corr:.3f} (p = {bin_spearman_p:.5f})")

        # ------------ VISUALIZATION ----------
        sns.regplot(x='similarity', y='cross_score', data=correlation_df, scatter_kws={'alpha': 0.3})
        try:
            sim_label = os.path.split(similarity_filepath)[-1].split('_', 1)[1].replace('.csv', '')
        except:
            sim_label = ""
        plt.title(f"{sim_label} Sim vs Cross Pipeline MCC")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Cross-Application Performance")
        plt.show()
    except Exception as e:
        print(f"ERROR during correlation analysis: {e}")


def compute_mann_whitney_u(similarity_filepath, cross_results_dir):
    """
    This function performs a Mann-Whitney U test to compare cross-application performance across different similarity groups. It includes:
    - Loading and preparing the similarity matrix and cross-application results
    - Grouping the data into low, medium, and high similarity categories
    - Performing one-sided Mann-Whitney U tests to compare performance between groups (e.g
        medium vs low, high vs low)
    - Interpreting the test results in terms of statistical significance and practical implications
    """
    similarity_df, correlation_df = load_and_prepare_similarity_and_cross_results(similarity_filepath,
                                                                                  cross_results_dir)

    # --- Step 6: Compute correlations for different similarity ranges
    low_similarity = correlation_df[correlation_df["similarity"] < 0.5]
    medium_similarity = correlation_df[(correlation_df["similarity"] >= 0.5) & (correlation_df["similarity"] < 0.7)]
    high_similarity = correlation_df[correlation_df["similarity"] >= 0.7]

    high = high_similarity["cross_score"]
    medium = medium_similarity["cross_score"]
    low = low_similarity["cross_score"]

    med_low_stat, med_low_p = mannwhitneyu(medium, low, alternative='greater')  # one-sided
    print(f"Mann-Whitney U-test p-value (Medium vs Low similarity): {med_low_p:.5f}")
    high_low_stat, high_low_p = mannwhitneyu(high, low, alternative='greater')  # one-sided
    print(f"Mann-Whitney U-test p-value (High vs Low similarity): {high_low_p:.5f}")
    print("Done.")


def bayesian_regression(similarity_filepath, cross_results_dir):
    """
    This function performs a Bayesian linear regression analysis to examine the relationship between dataset similarity and cross-application performance. It includes:
    - Setting up a Bayesian linear regression model with appropriate priors for the intercept, slope, and
        error term
    - Using Markov Chain Monte Carlo (MCMC) sampling to estimate the posterior distributions of the model parameters
    - Interpreting the posterior distributions to understand the strength and uncertainty of the relationship between similarity and
        cross-application performance, including credible intervals and probability statements about the direction of the effect.
    """
    similarity_df, correlation_df = load_and_prepare_similarity_and_cross_results(similarity_filepath,
                                                                                  cross_results_dir)
    x = correlation_df["similarity"].values
    y = correlation_df["cross_score"].values

    with pm.Model() as model:
        # Use Data for numpy arrays so PyMC can work with symbolic variables
        x_shared = pm.Data("x", x)

        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value - use standard arithmetic; pm.Data provides a symbolic array
        mu = alpha + beta * x_shared

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sampling
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    return model, trace


def interpret_bayesian(trace):
    summary = az.summary(trace, hdi_prob=0.95)

    beta_mean = summary.loc["beta", "mean"]
    hdi_low = summary.loc["beta", "hdi_2.5%"]
    hdi_high = summary.loc["beta", "hdi_97.5%"]

    interpretation = []

    interpretation.append(
        f"The posterior mean effect of similarity is {beta_mean:.4f}."
    )

    interpretation.append(
        f"The 95% credible interval is [{hdi_low:.4f}, {hdi_high:.4f}]."
    )

    if hdi_low > 0:
        interpretation.append("There is strong evidence that the effect is positive.")
    elif hdi_high < 0:
        interpretation.append("There is strong evidence that the effect is negative.")
    else:
        interpretation.append("The effect is uncertain and may be close to zero.")

    interpretation.append("...\n")

    return "\n<<<\n".join(interpretation)


def linear_regression_on_multiple_similarity_metrics(similarity_filepath, cross_results_dir, transfer_rate_threshold=0.5, top_n=20):
    """
    Insight analysis per pipeline (source):
    - Computes for each pipeline:
      * mean_cross_score
      * median_cross_score
      * transfer_rate (>0)
      * strong_transfer_rate (>0.1)
      * std_cross_score
      * min_cross_score
    - Filters pipelines with transfer_rate > transfer_rate_threshold
    - Ranks by mean_cross_score and by a combined score (mean * transfer_rate)

    Returns a dict with keys:
      - all: full dataframe of pipeline statistics
      - filtered: dataframe after applying transfer_rate filter
      - top_by_mean: top_n pipelines sorted by mean_cross_score
      - top_by_combined: top_n pipelines sorted by combined_score

    Parameters:
      similarity_filepath: path to similarity CSV (passed through helper loader)
      cross_results_dir: URL or path used by loader
      transfer_rate_threshold: float threshold to filter pipelines (default 0.5)
      top_n: number of top pipelines to return for the rankings
    """
    # Load data using existing helper
    try:
        _, correlation_df = load_and_prepare_similarity_and_cross_results(similarity_filepath, cross_results_dir)
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

    if correlation_df.empty:
        print("No cross application data was found (empty DataFrame).")
        empty_df = pd.DataFrame()
        return {
            'all': empty_df,
            'filtered': empty_df,
            'top_by_mean': empty_df,
            'top_by_combined': empty_df
        }

    # Ensure cross_score numeric
    correlation_df = correlation_df.copy()
    correlation_df['cross_score'] = pd.to_numeric(correlation_df['cross_score'], errors='coerce')

    # Drop rows with NaN cross_score
    correlation_df = correlation_df.dropna(subset=['cross_score'])

    # Group by source (pipeline)
    grouped = correlation_df.groupby('source')['cross_score']

    mean_cross = grouped.mean()
    median_cross = grouped.median()
    std_cross = grouped.std().fillna(0)
    min_cross = grouped.min()
    count_targets = grouped.count()
    transfer_rate = grouped.apply(lambda x: np.mean(x > 0))
    strong_transfer_rate = grouped.apply(lambda x: np.mean(x > 0.1))

    stats_df = pd.DataFrame({
        'mean_cross_score': mean_cross,
        'median_cross_score': median_cross,
        'std_cross_score': std_cross,
        'min_cross_score': min_cross,
        'count_targets': count_targets,
        'transfer_rate': transfer_rate,
        'strong_transfer_rate': strong_transfer_rate
    })

    # Combined score: simple and interpretable: mean * transfer_rate
    stats_df['combined_score'] = stats_df['mean_cross_score'] * stats_df['transfer_rate']

    # Add additional helpful columns
    stats_df['mean_per_target'] = stats_df['mean_cross_score']  # alias for readability

    # Filter
    filtered_df = stats_df[stats_df['transfer_rate'] > transfer_rate_threshold].copy()

    # Sortings
    top_by_mean = filtered_df.sort_values('mean_cross_score', ascending=False).head(top_n)
    top_by_combined = filtered_df.sort_values('combined_score', ascending=False).head(top_n)

    # Print short summary
    total_pipelines = len(stats_df)
    filtered_pipelines = len(filtered_df)
    print(f"Pipelines total: {total_pipelines}")
    print(f"Pipelines by transfer_rate > {transfer_rate_threshold}: {filtered_pipelines}")
    if not top_by_mean.empty:
        print("\nTop pipelines by mean_cross_score:")
        print(top_by_mean[['mean_cross_score', 'transfer_rate', 'combined_score']].head(10))
    if not top_by_combined.empty:
        print("\nTop pipelines by combined_score (mean * transfer_rate):")
        print(top_by_combined[['mean_cross_score', 'transfer_rate', 'combined_score']].head(10))

    # Return results for programmatic use
    return {
        'all': stats_df.sort_values('mean_cross_score', ascending=False),
        'filtered': filtered_df.sort_values('mean_cross_score', ascending=False),
        'top_by_mean': top_by_mean,
        'top_by_combined': top_by_combined
    }


# ==========================================
# Pipeline Reuse & Multi-Metric Analysis
# ==========================================

def load_cross_results_with_original_scores(cross_results_dir):
    """
    Load cross-application results from the GitHub directory and preserve:
      - source
      - target
      - original_score
      - cross_score

    Parameters
    ----------
    cross_results_dir : str
        GitHub API directory URL containing the *_pipeline.txt files.

    Returns
    -------
    pd.DataFrame
        Columns: source, target, original_score, cross_score
    """
    all_rows = []

    cross_results_files = read_files_from_url_or_folder(cross_results_dir)

    for file in cross_results_files:
        src_dataset = file["name"]

        if not src_dataset.endswith("_pipeline.txt"):
            continue

        raw_url = file["download_url"]
        df = pd.read_csv(raw_url, sep=';', engine='python')

        for _, row in df.iterrows():
            if len(row) < 4:
                continue

            if row.iloc[2] is None:
                warnings.warn(
                    f"Skipping empty row: {row.tolist()} in file: {src_dataset}",
                    UserWarning
                )
                continue

            # skip repeated headers inside file
            if (
                str(row.iloc[0]).strip() == 'Pipeline'
                and str(row.iloc[1]).strip() == 'OriginalScore'
                and str(row.iloc[2]).strip() == 'CrossApplication'
                and str(row.iloc[3]).strip() == 'CrossScore'
            ):
                continue

            try:
                tgt_dataset = normalize_name(str(row.iloc[2]))
                src_name = normalize_name(str(src_dataset))

                original_score = float(str(row.iloc[1]).replace(",", "."))
                cross_score = float(str(row.iloc[3]).replace(",", "."))
            except Exception as e:
                print(f"Skipping malformed row in {src_dataset}: {e} | row={row.tolist()}")
                continue

            all_rows.append({
                "source": src_name,
                "target": tgt_dataset,
                "original_score": original_score,
                "cross_score": cross_score
            })

    return pd.DataFrame(all_rows)


def load_multi_similarity_and_cross_results(similarity_filepaths, cross_results_dir):
    """
    Merge multiple pairwise similarity matrices with cross-application results.

    Parameters
    ----------
    similarity_filepaths : dict[str, str]
        Mapping {metric_name: filepath_to_similarity_csv}
    cross_results_dir : str
        GitHub API directory URL containing the *_pipeline.txt files.

    Returns
    -------
    dict
        {
            "similarity_dfs": dict[str, pd.DataFrame],
            "merged_df": pd.DataFrame
        }

    Notes
    -----
    - Similarity matrices must have dataset names as both index and columns.
    - Names are matched via LONG_TO_SHORT_NAME using normalized source/target names.
    """
    cross_df = load_cross_results_with_original_scores(cross_results_dir)

    if cross_df.empty:
        return {
            "similarity_dfs": {},
            "merged_df": cross_df
        }

    similarity_dfs = {}
    merged_df = cross_df.copy()

    for metric_name, filepath in similarity_filepaths.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            sim_df = pd.read_csv(
                filepath,
                index_col=0,
                usecols=lambda col: col != filepath.split('/')[-1]
            )

        sim_df.columns = sim_df.columns.astype(str).str.strip()
        sim_df.index = sim_df.index.astype(str).str.strip()
        similarity_dfs[metric_name] = sim_df

        values = []
        for _, row in merged_df.iterrows():
            src = row["source"]
            tgt = row["target"]

            try:
                src_key = LONG_TO_SHORT_NAME[src]
                tgt_key = LONG_TO_SHORT_NAME[tgt]
                sim_val = sim_df.loc[src_key, tgt_key]
                sim_val = float(str(sim_val).replace(",", "."))
            except Exception:
                sim_val = np.nan

            values.append(sim_val)

        merged_df[metric_name] = values

    return {
        "similarity_dfs": similarity_dfs,
        "merged_df": merged_df
    }


def compute_source_transfer_statistics(cross_results_dir, strong_transfer_threshold=0.1):
    """
    Compute source-pipeline level transfer statistics.

    Parameters
    ----------
    cross_results_dir : str
        GitHub API directory URL containing the *_pipeline.txt files.
    strong_transfer_threshold : float, default=0.1
        Threshold above which a transfer is considered 'strong'.

    Returns
    -------
    pd.DataFrame
        Indexed by source dataset / pipeline with columns:
        - mean_cross_score
        - median_cross_score
        - std_cross_score
        - min_cross_score
        - max_cross_score
        - mean_original_score
        - mean_drop
        - retention_mean
        - transfer_rate
        - strong_transfer_rate
        - count_targets
        - combined_score
    """
    cross_df = load_cross_results_with_original_scores(cross_results_dir)

    if cross_df.empty:
        return pd.DataFrame()

    df = cross_df.copy()
    df["score_drop"] = df["original_score"] - df["cross_score"]

    # ratio can explode when original_score is close to zero
    df["retention"] = np.where(
        np.abs(df["original_score"]) > 1e-9,
        df["cross_score"] / df["original_score"],
        np.nan
    )
    df["positive_transfer"] = (df["cross_score"] > 0).astype(int)
    df["strong_transfer"] = (df["cross_score"] > strong_transfer_threshold).astype(int)

    grouped = df.groupby("source")

    out = grouped.agg(
        mean_cross_score=("cross_score", "mean"),
        median_cross_score=("cross_score", "median"),
        std_cross_score=("cross_score", "std"),
        min_cross_score=("cross_score", "min"),
        max_cross_score=("cross_score", "max"),
        mean_original_score=("original_score", "mean"),
        mean_drop=("score_drop", "mean"),
        retention_mean=("retention", "mean"),
        transfer_rate=("positive_transfer", "mean"),
        strong_transfer_rate=("strong_transfer", "mean"),
        count_targets=("target", "count")
    ).reset_index()

    out["std_cross_score"] = out["std_cross_score"].fillna(0.0)

    # A simple ranking score balancing quality + robustness
    out["combined_score"] = (
        out["mean_cross_score"] * out["transfer_rate"]
        - 0.25 * out["std_cross_score"]
    )

    return out.sort_values("combined_score", ascending=False)


def compare_single_metric_ols(similarity_filepaths, cross_results_dir, standardize=True):
    """
    Fit one OLS model per similarity metric:
        cross_score ~ metric

    Parameters
    ----------
    similarity_filepaths : dict[str, str]
        Mapping {metric_name: filepath}
    cross_results_dir : str
        GitHub API directory URL
    standardize : bool, default=True
        Z-standardize predictor before regression.

    Returns
    -------
    dict
        {
            "models": dict[str, statsmodels.regression.linear_model.RegressionResultsWrapper],
            "summary_df": pd.DataFrame,
            "merged_df": pd.DataFrame
        }
    """
    loaded = load_multi_similarity_and_cross_results(similarity_filepaths, cross_results_dir)
    df = loaded["merged_df"].copy()

    models = {}
    rows = []

    for metric_name in similarity_filepaths.keys():
        tmp = df[["cross_score", metric_name]].dropna().copy()

        if len(tmp) < 5:
            print(f"Skipping {metric_name}: not enough valid observations.")
            continue

        x = tmp[metric_name].astype(float)
        if standardize:
            x_std = x.std(ddof=0)
            if x_std > 0:
                x = (x - x.mean()) / x_std

        X = sm.add_constant(x)
        y = tmp["cross_score"].astype(float)

        model = sm.OLS(y, X).fit()
        models[metric_name] = model

        coef_name = metric_name if metric_name in model.params.index else model.params.index[-1]

        rows.append({
            "metric": metric_name,
            "n_obs": int(model.nobs),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "coef": model.params[coef_name],
            "p_value": model.pvalues[coef_name],
            "aic": model.aic,
            "bic": model.bic
        })

    summary_df = pd.DataFrame(rows).sort_values("r_squared", ascending=False)

    return {
        "models": models,
        "summary_df": summary_df,
        "merged_df": df
    }


def compute_similarity_metric_correlations(similarity_filepaths, cross_results_dir, method="pearson"):
    """
    Compute pairwise correlations between similarity metrics.

    Parameters
    ----------
    similarity_filepaths : dict[str, str]
        Mapping {metric_name: filepath}
    cross_results_dir : str
        GitHub API directory URL
    method : str, default='pearson'
        Correlation method for pandas.DataFrame.corr

    Returns
    -------
    pd.DataFrame
        Correlation matrix among similarity metrics.
    """
    loaded = load_multi_similarity_and_cross_results(similarity_filepaths, cross_results_dir)
    df = loaded["merged_df"].copy()

    metric_cols = list(similarity_filepaths.keys())
    metric_df = df[metric_cols].apply(pd.to_numeric, errors="coerce")

    return metric_df.corr(method=method)


def fit_combined_ols_model(
    similarity_filepaths,
    cross_results_dir,
    include_original_score=True,
    standardize_predictors=True,
    drop_high_correlation_above=None
):
    """
    Fit one combined OLS model:
        cross_score ~ metric_1 + metric_2 + ... (+ original_score)

    Parameters
    ----------
    similarity_filepaths : dict[str, str]
        Mapping {metric_name: filepath}
    cross_results_dir : str
        GitHub API directory URL
    include_original_score : bool, default=True
        Whether to include original_score as predictor.
    standardize_predictors : bool, default=True
        Z-standardize all predictors.
    drop_high_correlation_above : float or None, default=None
        If set (e.g. 0.85), greedily drops later predictors that are
        highly correlated with earlier ones.

    Returns
    -------
    dict
        {
            "model": statsmodels regression result,
            "used_predictors": list[str],
            "dropped_predictors": list[str],
            "dataframe": pd.DataFrame
        }
    """
    loaded = load_multi_similarity_and_cross_results(similarity_filepaths, cross_results_dir)
    df = loaded["merged_df"].copy()

    predictor_cols = list(similarity_filepaths.keys())
    if include_original_score:
        predictor_cols.append("original_score")

    work_df = df[["cross_score"] + predictor_cols].copy()
    work_df = work_df.apply(pd.to_numeric, errors="coerce").dropna()

    used_predictors = predictor_cols.copy()
    dropped_predictors = []

    if drop_high_correlation_above is not None and len(used_predictors) > 1:
        corr = work_df[used_predictors].corr().abs()
        keep = []
        for col in used_predictors:
            if not keep:
                keep.append(col)
                continue
            too_similar = any(corr.loc[col, kept] > drop_high_correlation_above for kept in keep)
            if too_similar:
                dropped_predictors.append(col)
            else:
                keep.append(col)
        used_predictors = keep

    X = work_df[used_predictors].copy()

    if standardize_predictors:
        for col in X.columns:
            col_std = X[col].std(ddof=0)
            if col_std > 0:
                X[col] = (X[col] - X[col].mean()) / col_std
            else:
                X[col] = 0.0

    X = sm.add_constant(X)
    y = work_df["cross_score"]

    model = sm.OLS(y, X).fit()

    return {
        "model": model,
        "used_predictors": used_predictors,
        "dropped_predictors": dropped_predictors,
        "dataframe": work_df
    }


def fit_transfer_logistic_model(
    similarity_filepaths,
    cross_results_dir,
    good_transfer_threshold=0.05,
    include_original_score=True,
    standardize_predictors=True
):
    """
    Fit a logistic regression model:
        P(good_transfer) ~ metric_1 + metric_2 + ... (+ original_score)

    where:
        good_transfer = 1 if cross_score > good_transfer_threshold else 0

    Parameters
    ----------
    similarity_filepaths : dict[str, str]
        Mapping {metric_name: filepath}
    cross_results_dir : str
        GitHub API directory URL
    good_transfer_threshold : float, default=0.05
        Threshold for positive / useful transfer.
    include_original_score : bool, default=True
        Whether to include original_score.
    standardize_predictors : bool, default=True
        Z-standardize predictors.

    Returns
    -------
    dict
        {
            "model": statsmodels LogitResults,
            "dataframe": pd.DataFrame,
            "class_balance": float
        }
    """
    loaded = load_multi_similarity_and_cross_results(similarity_filepaths, cross_results_dir)
    df = loaded["merged_df"].copy()

    predictor_cols = list(similarity_filepaths.keys())
    if include_original_score:
        predictor_cols.append("original_score")

    work_df = df[["cross_score"] + predictor_cols].copy()
    work_df = work_df.apply(pd.to_numeric, errors="coerce").dropna()

    work_df["good_transfer"] = (work_df["cross_score"] > good_transfer_threshold).astype(int)

    X = work_df[predictor_cols].copy()
    if standardize_predictors:
        for col in X.columns:
            col_std = X[col].std(ddof=0)
            if col_std > 0:
                X[col] = (X[col] - X[col].mean()) / col_std
            else:
                X[col] = 0.0

    X = sm.add_constant(X)
    y = work_df["good_transfer"]

    # suppress common convergence chatter; user can inspect summary afterward
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y, X).fit(disp=False, maxiter=200)

    return {
        "model": model,
        "dataframe": work_df,
        "class_balance": float(y.mean())
    }


def summarize_target_dataset_difficulty(cross_results_dir, strong_transfer_threshold=0.1):
    """
    Summarize how 'difficult' each target dataset is as a transfer destination.

    Parameters
    ----------
    cross_results_dir : str
        GitHub API directory URL
    strong_transfer_threshold : float, default=0.1
        Threshold for strong incoming transfer.

    Returns
    -------
    pd.DataFrame
        Indexed by target dataset with columns such as:
        - incoming_mean_cross_score
        - incoming_median_cross_score
        - incoming_std_cross_score
        - incoming_min_cross_score
        - incoming_transfer_rate
        - incoming_strong_transfer_rate
        - n_sources
    """
    cross_df = load_cross_results_with_original_scores(cross_results_dir)

    if cross_df.empty:
        return pd.DataFrame()

    df = cross_df.copy()
    df["incoming_positive_transfer"] = (df["cross_score"] > 0).astype(int)
    df["incoming_strong_transfer"] = (df["cross_score"] > strong_transfer_threshold).astype(int)

    grouped = df.groupby("target")

    out = grouped.agg(
        incoming_mean_cross_score=("cross_score", "mean"),
        incoming_median_cross_score=("cross_score", "median"),
        incoming_std_cross_score=("cross_score", "std"),
        incoming_min_cross_score=("cross_score", "min"),
        incoming_max_cross_score=("cross_score", "max"),
        incoming_transfer_rate=("incoming_positive_transfer", "mean"),
        incoming_strong_transfer_rate=("incoming_strong_transfer", "mean"),
        n_sources=("source", "count")
    ).reset_index()

    out["incoming_std_cross_score"] = out["incoming_std_cross_score"].fillna(0.0)

    # easier datasets at the top
    return out.sort_values(
        ["incoming_mean_cross_score", "incoming_transfer_rate"],
        ascending=False
    )


def correlate_target_difficulty_with_dataset_metrics(target_metrics_df, target_difficulty_df, method="spearman"):
    """
    Correlate target-dataset difficulty summaries with dataset-level complexity metrics.

    Parameters
    ----------
    target_metrics_df : pd.DataFrame
        Dataset-level metrics indexed by dataset name.
        Example columns:
            edge_density, texture_features, hist_entropy, fourier_frequency, num_superpixels
    target_difficulty_df : pd.DataFrame
        Output of summarize_target_dataset_difficulty(...)
    method : str, default='spearman'
        'pearson' or 'spearman'

    Returns
    -------
    pd.DataFrame
        Long-format correlation table with columns:
            difficulty_metric, dataset_metric, correlation, p_value, n
    """
    if target_metrics_df is None or target_metrics_df.empty:
        raise ValueError("target_metrics_df is empty.")

    if target_difficulty_df is None or target_difficulty_df.empty:
        raise ValueError("target_difficulty_df is empty.")

    metrics_df = target_metrics_df.copy()
    metrics_df.index = metrics_df.index.astype(str).str.strip()

    difficulty_df = target_difficulty_df.copy()
    difficulty_df["target"] = difficulty_df["target"].astype(str).str.strip()

    merged = difficulty_df.merge(
        metrics_df,
        left_on="target",
        right_index=True,
        how="inner"
    )

    difficulty_cols = [
        "incoming_mean_cross_score",
        "incoming_median_cross_score",
        "incoming_std_cross_score",
        "incoming_transfer_rate",
        "incoming_strong_transfer_rate"
    ]
    dataset_metric_cols = [c for c in metrics_df.columns]

    rows = []

    for dcol in difficulty_cols:
        for mcol in dataset_metric_cols:
            tmp = merged[[dcol, mcol]].apply(pd.to_numeric, errors="coerce").dropna()

            if len(tmp) < 3:
                corr, p_val = np.nan, np.nan
            else:
                if method.lower() == "pearson":
                    corr, p_val = pearsonr(tmp[dcol], tmp[mcol])
                else:
                    corr, p_val = spearmanr(tmp[dcol], tmp[mcol])

            rows.append({
                "difficulty_metric": dcol,
                "dataset_metric": mcol,
                "correlation": corr,
                "p_value": p_val,
                "n": len(tmp)
            })

    return pd.DataFrame(rows).sort_values(
        ["difficulty_metric", "p_value"],
        ascending=[True, True]
    )


def perform_pipeline_reuse_multimetric_analysis(
        similarity_files,
        cross_results_dir):
    # 1. Pipeline-level analysis
    stats = compute_source_transfer_statistics(cross_results_dir)
    print(stats.head())

    # 2. Single metric OLS
    single = compare_single_metric_ols(similarity_files, cross_results_dir)
    print(single["summary_df"])

    # 3. Combined OLS
    combined = fit_combined_ols_model(similarity_files, cross_results_dir)
    print(combined["model"].summary())

    # 4. Logistic model
    logit = fit_transfer_logistic_model(similarity_files, cross_results_dir)
    print(logit["model"].summary())
