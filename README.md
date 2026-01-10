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
taskRetrieval/
├── main.py                # Main entry point to run the pipeline
├── experiment_params_data.py  # Dataset configurations & experiment settings
├── env_vars.py            # Environment variables & file paths
├── README.md              # Project documentation
├── results/               # Output directory for experiment results
├── datasets/              # Input datasets
├── models/                # Model definitions
└── scripts/               # Scripts for training, evaluation, and utilities
```

## 🚀 Approach Overview
The project pipeline analyzes segmentation datasets by extracting feature embeddings, 
computing similarities, and evaluating 
segmentation performance statistically.

#### 🔧 Steps:
* Convert images to grayscale
* Extract feature embeddings using a ResNet model
* Compute the cosine similarity between feature vectors
* Cross-Apply filter pipelines (CGP) to datasets
* Compute the spearman corellation between dataset features and segmentation performance

### 🧩 1. Feature Extraction
Goal: Represent each dataset through a vector of statistical and visual features.

Use a pre-trained CNN (e.g., **ResNet-18**, **ResNet-50**, or **VGG-16**) to embed each image into a fixed-dimensional vector space.

For example, remove the final classification layer and use the **penultimate layer** (e.g., a 512-dimensional vector for ResNet-18).

Let each image  
&nbsp;&nbsp;&nbsp;&nbsp;_x_ ∈ ℝ<sup>H×W×3</sup>  
be mapped to an embedding  
&nbsp;&nbsp;&nbsp;&nbsp;_f(x)_ ∈ ℝ<sup>d</sup>.


### 🔍 3. Similarity Computation
Goal: Measure pairwise similarity within each dataset cluster.

* Compute cosine similarity between dataset feature vectors
* Cosine Similarity:

&nbsp;&nbsp;&nbsp;&nbsp;_cosine(x, y) = (x ⋅ y) / (||x|| ||y||)_

#### Interpretation
* Empirically in computer vision, cosine similarities above `> 0.7` are typically considered "highly similar" in embedding space.
* `0.5 – 0.6` is moderate, but could be meaningful if your downstream task (CGP pipeline transfer) is sensitive to partial overlap in features.
* `< 0.5` usually indicates the datasets differ enough that you wouldn’t expect strong cross-domain generalization.

### 📏 4. MCC Score Computation
Goal: Evaluate segmentation performance on selected dataset pairs.

* Cross-Apply CGP pipelines between each pair of datasets
* Compute the Matthews Correlation Coefficient (MCC) for each pair

&nbsp;&nbsp;&nbsp;&nbsp;_MCC = (TP × TN - FP × FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))_

Where:
* TP = True Positives
* TN = True Negatives
* FP = False Positives
* FN = False Negatives

### 📊 🧠 Statistical Hypothesis
* Null Hypothesis (H0): There is no correlation between dataset similarity and cross-application performance.
* Alternative Hypothesis (H1): Higher similarity does lead to better cross-application performance.

_If the p-value < 0.05, you reject H0 and conclude that the correlation is statistically significant._

#### Spearman Correlation

* Start with Spearman correlation (rank-based, non-parametric, robust to non-linear relationships):

&nbsp;&nbsp;&nbsp;&nbsp;_ρ = 1 - (6 Σ d<sub>i</sub><sup>2</sup>) / (n(n<sup>2</sup> - 1))_

Where:
* _d<sub>i</sub>_ = difference between ranks of each pair
* _n_ = number of pairs
* ρ = Spearman correlation coefficient (ranges from -1 to 1)
* If ρ > 0, it indicates a positive correlation between similarity and performance.
* If ρ < 0, it indicates a negative correlation.
* If |ρ| is close to 1, it indicates a strong correlation.
* If |ρ| is close to 0, it indicates a weak correlation.
* If p < 0.05, the correlation is statistically significant.
* If p > 0.05, the correlation is not statistically significant.
 

#### Linear Regression

Fit a simple regression model to quantify how much similarity influences performance:

&nbsp;&nbsp;&nbsp;&nbsp;_y = β<sub>0</sub> + β<sub>1</sub>x + ε_

_Where:_
* _y_ = MCC score (segmentation performance)
* _x_ = similarity score (cosine similarity)
* β<sub>0</sub> = intercept
* β<sub>1</sub> = slope (how much performance changes with similarity)
* ε = error term (residuals)

#### Stratified Group Comparison
Divide dataset pairs into e.g. 3 bins:
* High similarity (top 33%)
* Medium similarity
* Low similarity (bottom 33%)

Then run Mann-Whitney U-tests (non-parametric) between performance distributions of the groups:

*If p < 0.05, high similarity groups statistically outperform low similarity ones.*

### 🎯 Scientific Goals

* 📚 Understand Dataset Similarity

Gain insight into structural and visual similarities across datasets.

* 🧪 Evaluate Segmentation Performance: Measure how performance varies with dataset similarity.

* 📐 Validate Statistically: Use robust methods (correlation, Mann-Whitney-U) to confirm hypotheses.

## 📖 How to Cite
If you use this code or ideas from our paper, please cite:

```bibtex
@misc{margraf2025have,
 title   = {Have I Solved This Before? Retrieving Similar Segmentation Problems for Model Training},
 author  = {Margraf, Andreas and Cui, Henning and Haehner, Joerg},
 year    = {2025},
 eprint  = {0000.00000},
 archivePrefix  = {arXiv},
 primaryClass  = {cs.CS},
 url     = {https://arxiv.org/abs/0000.00000}
}
```