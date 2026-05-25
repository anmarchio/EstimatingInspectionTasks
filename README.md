# Retrieving Similar Segmentation Problems for Evolutionary Learning

This repository contains the code and scripts for the paper  
**_"Have I Solved This Before? Retrieving Similar Segmentation Problems for Evolutionary Learning"_**

---

## 🧾 Abstract

This study proposes a novel approach to improve the efficiency of segmentation model training by leveraging previously solved segmentation problems. Instead of focusing solely on algorithm design, this method emphasizes **understanding the problem domain** through dataset analysis.

🔍 **Key Ideas**:
- Modern production systems demand self-adaptive, self-configuring software.
- Early-stage design is often done under uncertainty, limiting downstream flexibility.
- The approach focuses on understanding and **comparing datasets**, enabling reuse of prior configurations.
- A **centralized knowledge base** evolves over time, supporting retrieval of similar solutions.
- **Model reuse** reduces training effort, shortens development time, and avoids late-stage redesign.
- Simple, well-fitted models are favored to balance complexity, reliability, and resource constraints.

---

## 📁 Project Structure

```bash
EstimatingInspectionTasks/
├── data/                  # context data for potential model fitting
├── results/               # Output directory for experiment results
├── src/                   # source code for feature extraction, similarity computation, and analysis
├── .python-version.py     # Sets a specific python version this repo is created for
├── env_vars.py            # Environment variables & file paths
├── experiment_params_data.py  # Dataset configurations & experiment settings
├── LICENSE                # license information
├── main.py                # Main entry point to run the pipeline
├── README.md              # Project documentation
├── requirements.txt       # requirements file
```

## Approach Overview
Previous to this study, research has been conducted on image filter pipelines evolved by mean of CGP:
* ACSOS 2023
* Dissertation 2026

In Follow-up experiments different complexity image features are extracted from segmentation datasets 
These are utilized to compute similarities and evaluate cross-application (from dataset i to dataset j) 
performance statistically.

Overall, the experimentation was conducted in the following order:

* Compute similarity metrics between datasets (labelled image data collection, cf. Margraf 2023)
* Compute the cosine similarity between feature vectors
* Cross-Apply filter pipelines (CGP) to datasets, see project:
* Perform statistical analysis:
  * correlation analysis between dataset features and cross-pipeline performance
  * single-metric OLS analysis
  * multi-metric OLS model
  * p-value and R^2 significance evaluation
    
## Statistical Analysis Overview
The statistical analysis investigates whether image and dataset similarity can predict the cross-application transfer performance of CGP-based segmentation pipelines.

For this purpose, multiple image complexity and structural descriptors were extracted, including:
- CNN (pretrained ResNet-50) embedding similarity
- Histogram entropy
- Texture similarity (GLCM)
- Edge density
- Fourier frequency characteristics
- Number of superpixels

Pairwise dataset similarity was computed using cosine similarity:

\[
C_{sim}(A,B)=\frac{A \cdot B}{\|A\|\|B\|}
\]

where \(A\) and \(B\) represent feature vectors of two datasets.

The relationship between similarity and cross-application performance (\(MCC\)) was analyzed using several statistical methods.

### Correlation Analysis

Linear and monotonic dependencies were analyzed using Pearson and Spearman correlation coefficients:

\[
r=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}}
\]

\[
\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}
\]

The analysis was performed globally and additionally within multiple similarity bins to evaluate local transfer behavior at different similarity ranges.

### Ordinary Least Squares (OLS) Regression

Linear regression models were used to estimate the influence of similarity metrics on transfer performance:

\[
y = \beta_0 + \beta_1 x + \epsilon
\]

where:
- \(x\) denotes similarity,
- \(y\) denotes cross-application MCC,
- \(\beta_1\) represents the transfer effect.

Model quality was evaluated using:

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

Residual diagnostics included:
- Durbin-Watson test
- Jarque-Bera test
- Omnibus normality test

### Bayesian Regression

Bayesian linear regression was applied to estimate posterior distributions of transfer effects:

\[
y_i \sim \mathcal{N}(\mu_i,\sigma)
\]

\[
\mu_i = \alpha + \beta x_i
\]

with priors:

\[
\alpha,\beta \sim \mathcal{N}(0,1)
\]

\[
\sigma \sim \text{HalfNormal}(1)
\]

Posterior estimation was performed using MCMC sampling.

### Mann-Whitney U Test

To compare transferability between low-, medium-, and high-similarity groups, a one-sided Mann-Whitney U test was applied:

\[
U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1
\]

### Transferability Analysis

In addition to regression and correlation analysis, pipeline transferability statistics were computed, including:
- Mean cross-application MCC
- Median MCC
- Transfer rate
- Strong transfer rate
- Combined transfer score

Pipelines were ranked according to transfer performance and robustness across datasets.

A detailed overview of all statistical results, correlations, regression summaries, transfer scores, and visualizations is provided in `Results.md`. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## Detailed Results

See: [Results.md](Results.md).

## 📖 How to Cite
If you use this code or ideas from our paper, please cite:

```bibtex
@misc{margraf2026have,
 title   = {Have I Solved This Before? Retrieving Similar Segmentation Problems for Evolutionary Learning},
 author  = {Margraf, Andreas and Cui, Henning and Haehner, Joerg},
 year    = {2026},
 eprint  = {0000.00000},
 archivePrefix  = {arXiv},
 primaryClass  = {cs.CS},
 url     = {https://arxiv.org/abs/0000.00000}
}
```