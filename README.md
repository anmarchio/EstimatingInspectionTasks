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

C_sim(A,B) = (A · B) / (||A|| ||B||)

where A and B represent feature vectors of two datasets.

### Correlation Analysis

Linear and monotonic dependencies between similarity and cross-application performance (MCC) were analyzed using Pearson and Spearman correlation coefficients.

Pearson correlation:

r = Σ((xᵢ - x̄)(yᵢ - ȳ)) / √(Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)²)

Spearman rank correlation:

ρ = 1 - (6 Σdᵢ²) / (n(n² - 1))

The analysis was performed globally and additionally within multiple similarity bins to evaluate local transfer behavior at different similarity ranges.

### Ordinary Least Squares (OLS) Regression

Linear regression models were used to estimate the influence of similarity metrics on transfer performance:

y = β₀ + β₁x + ε

where:
- x denotes similarity,
- y denotes cross-application MCC,
- β₁ represents the transfer effect.

Model quality was evaluated using:

R² = 1 - (SS_res / SS_tot)

Residual diagnostics included:
- Durbin-Watson test
- Jarque-Bera test
- Omnibus normality test

## Transferability Analysis

In addition to regression and correlation analysis, pipeline transferability statistics were computed, including:
- Mean cross-application MCC
- Median MCC
- Transfer rate
- Strong transfer rate
- Combined transfer score

Pipelines were ranked according to transfer performance and robustness across datasets.

Detailed numerical results, regression summaries, transfer statistics, similarity-bin analyses, and visualizations are provided in `Results.md`.
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