import arviz as az
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import requests
import seaborn as sns
import statsmodels.api as sm
import warnings
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr, pearsonr
from skimage.feature import graycomatrix, graycoprops
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest

from env_vars import LONG_TO_SHORT_NAME


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


def normalize_name(s):
    suffixes = ["_mean_pipeline", "_best_pipeline", ".txt"]
    for suffix in suffixes:
        s = s.replace(suffix, "")
    return s.strip()


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

    # --- Step 3: Load each result file and collect scores
    response = requests.get(cross_results_dir)
    cross_results_files = [item for item in response.json() if item["type"] == "file"]

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

            try:
                similarity = similarity_df.loc[
                    LONG_TO_SHORT_NAME[src_dataset],
                    LONG_TO_SHORT_NAME[tgt_dataset]
                ]
            except Exception as e:
                print(f"Skipping pair due to error: {e} in file: {src_dataset} with target: {tgt_dataset}")
                continue

            all_rows.append({
                "source": src_dataset,
                "target": tgt_dataset,
                "similarity": similarity,
                "cross_score": float(row.iloc[3])
            })

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

    # --- Step 5: Compute correlations
    pearson_corr, pearson_p = pearsonr(correlation_df["similarity"], correlation_df["cross_score"])
    spearman_corr, spearman_p = spearmanr(correlation_df["similarity"], correlation_df["cross_score"])

    print("✅ Pearson correlation:", pearson_corr, " (p =", pearson_p, ")")
    print("✅ Spearman correlation:", spearman_corr, " (p =", spearman_p, ")")

    # --- Step 6: Compute correlations for different similarity ranges
    low_similarity = correlation_df[correlation_df["similarity"] < 0.5]
    medium_similarity = correlation_df[(correlation_df["similarity"] >= 0.5) & (correlation_df["similarity"] < 0.7)]
    high_similarity = correlation_df[correlation_df["similarity"] >= 0.7]

    # Low similarity
    low_pearson_corr, low_pearson_p = pearsonr(low_similarity["similarity"], low_similarity["cross_score"])
    low_spearman_corr, low_spearman_p = spearmanr(low_similarity["similarity"], low_similarity["cross_score"])
    print("✅ Low Similarity - Pearson correlation:", low_pearson_corr, " (p =", low_pearson_p, ")")
    print("✅ Low Similarity - Spearman correlation:", low_spearman_corr, " (p =", low_spearman_p, ")")

    # Medium similarity
    medium_pearson_corr, medium_pearson_p = pearsonr(medium_similarity["similarity"], medium_similarity["cross_score"])
    medium_spearman_corr, medium_spearman_p = spearmanr(medium_similarity["similarity"],
                                                        medium_similarity["cross_score"])
    print("✅ Medium Similarity - Pearson correlation:", medium_pearson_corr, " (p =", medium_pearson_p, ")")
    print("✅ Medium Similarity - Spearman correlation:", medium_spearman_corr, " (p =", medium_spearman_p, ")")

    # High similarity
    high_pearson_corr, high_pearson_p = pearsonr(high_similarity["similarity"], high_similarity["cross_score"])
    high_spearman_corr, high_spearman_p = spearmanr(high_similarity["similarity"], high_similarity["cross_score"])
    print("✅ High Similarity - Pearson correlation:", high_pearson_corr, " (p =", high_pearson_p, ")")
    print("✅ High Similarity - Spearman correlation:", high_spearman_corr, " (p =", high_spearman_p, ")")

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
    plt.title("Similarity vs Pipeline Transfer Performance")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Cross-Application Performance")
    plt.show()


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

def deep_linear_regression(similarity_filepath, cross_results_dir, transfer_rate_threshold=0.5, top_n=20):
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
    print(f"Pipelines insgesamt: {total_pipelines}")
    print(f"Pipelines mit transfer_rate > {transfer_rate_threshold}: {filtered_pipelines}")
    if not top_by_mean.empty:
        print("\nTop pipelines nach mean_cross_score:")
        print(top_by_mean[['mean_cross_score', 'transfer_rate', 'combined_score']].head(10))
    if not top_by_combined.empty:
        print("\nTop pipelines nach combined_score (mean * transfer_rate):")
        print(top_by_combined[['mean_cross_score', 'transfer_rate', 'combined_score']].head(10))

    # Return results for programmatic use
    return {
        'all': stats_df.sort_values('mean_cross_score', ascending=False),
        'filtered': filtered_df.sort_values('mean_cross_score', ascending=False),
        'top_by_mean': top_by_mean,
        'top_by_combined': top_by_combined
    }
