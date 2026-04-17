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

### 🔧 Steps:
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


### 🔍 2. Similarity Computation
Goal: Measure pairwise similarity within each dataset cluster.

* Compute cosine similarity between dataset feature vectors
* Cosine Similarity:

&nbsp;&nbsp;&nbsp;&nbsp;_cosine(x, y) = (x ⋅ y) / (||x|| ||y||)_

#### Interpretation
* Empirically in computer vision, cosine similarities above `> 0.7` are typically considered "highly similar" in embedding space.
* `0.5 – 0.6` is moderate, but could be meaningful if your downstream task (CGP pipeline transfer) is sensitive to partial overlap in features.
* `< 0.5` usually indicates the datasets differ enough that you wouldn’t expect strong cross-domain generalization.

### 📏 3. MCC Score Computation
Goal: Evaluate segmentation performance on selected dataset pairs.

* Cross-Apply CGP pipelines between each pair of datasets
* Compute the Matthews Correlation Coefficient (MCC) for each pair

&nbsp;&nbsp;&nbsp;&nbsp;_MCC = (TP × TN - FP × FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))_

Where:
* TP = True Positives
* TN = True Negatives
* FP = False Positives
* FN = False Negatives

### 📐 4. Spearman Correlation

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
 

### 📚 5. Linear Regression (OLS)

Fit a simple regression model to quantify how much similarity influences performance:

&nbsp;&nbsp;&nbsp;&nbsp;_y = β<sub>0</sub> + β<sub>1</sub>x + ε_

_Where:_
* _y_ = MCC score (segmentation performance)
* _x_ = similarity score (cosine similarity)
* β<sub>0</sub> = intercept
* β<sub>1</sub> = slope (how much performance changes with similarity)
* ε = error term (residuals)

### 🧠 x. Statistical Hypothesis
* Null Hypothesis (H0): There is no correlation between dataset similarity and cross-application performance.
* Alternative Hypothesis (H1): Higher similarity does lead to better cross-application performance.

_If the p-value < 0.05, you reject H0 and conclude that the correlation is statistically significant._

### [---] Stratified Group Comparison (DROPPED!)
Divide dataset pairs into e.g. 3 bins:
* High similarity (top 33%)
* Medium similarity
* Low similarity (bottom 33%)

Then run Mann-Whitney U-tests (non-parametric) between performance distributions of the groups:

*If p < 0.05, high similarity groups statistically outperform low similarity ones.*

## 🧪 In-Depth Analyses
👉👉👉 **T O D O:**
- [ ] Multi-dimensional Similarity
- [ ] comparing weak against strong pipelines
- [ ] asymmetry
- [ ] cluster datasets by transfer behavior (not similarity)
- [ ] analyze failure modes
- [ ] include additional predictors (original_score, dataset type, pipeline complexity)
- [ ] Suggest / Try a reuse model

### 📊 Multi-dimensional Similarity
In addition to traditional OLS regression and correlation, we can also test specific hypotheses about the relationship between dataset similarity and segmentation performance. This allows a multi-dimensional insight.

Use several similarity measures to capture different aspects of dataset similarity:
- CNN embeddings
- [ ] jpeg_complexity?
- [ ] hist_entropy
- [ ] texture_features
- [ ] edge_density
- [ ] number_of_superpixels
- [ ] fourier_frequency

👉Then fit a regression model for each.

#### NEXT: Reusability model

* Mean transfer score: `mean_transfer_score = mean(cross_score / original_score)`
* Median transfer score: `median_transfer_score = median(cross_score / original_score)`
  * robust against outliers (important in your data)

* Transfer rate: `transfer_rate = #(cross_score > 0) / total`
  * measures how often the pipeline works at all

* Strong transfer rate: `strong_transfer_rate = #(cross_score > 0.1) / total`
  * measures how often the pipeline works well

* Worst-case (risk): `worst_case = min(cross_score)`
  * measures the potential downside of reuse

* Variacnce /std: `transfer_variance = var(cross_score)`
  * measures stability vs brittleness

#### 🔥 Best practical setup

For each pipeline - Compute:
* mean_cross_score
* median_cross_score
* transfer_rate (>0)
* strong_transfer_rate (>0.1)
* std_cross_score
* min_cross_score

Then:
* 👉 Filter: transfer_rate > 0.5
* 👉 Rank by:
  * mean_cross_score
  * OR combined score

### 📚 Comparing weak against strong pipelines

Positive transfer frequency:

`transfer rate= #(cross_score > 0) / total`

👉 Pipelines with:
* many small positive scores
* better than pipelines with few extreme highs

Hypothesis: **Overfitted strong pipelines are worse starting points than moderate ones.**

💡Introduce _transfer score_ to normalize performance:

`transfer_score = cross_score / original_score`

* Removes bias from strong vs weak pipelines
* Measures generalization efficiency
* 👉 __Likely more informative than plain MCC.__

### 🧪 Asymetry

Analyze asymmetry by transfer score in transfer performance matrix:

`T(A → B) - T(B → A)`

👉 __Rank by `|T(A → B) - T(B → A)|` to identify pairs with strong asymmetry.__

### 🧬 Cluster datasets by transfer behavior (not similarity)

Instead of your similarity matrix:

Build new feature vector per dataset:
* vector of cross-scores to all other datasets

Then:
* hierarchical clustering / spectral clustering

👉 This gives:
* “functional similarity” instead of visual similarity

💡 This is likely a publishable insight:
* similarity in transfer space ≠ similarity in feature space

### 📉 Analyze failure modes

Split into:
* successful transfer (MCC > 0.1)
* failed transfer (MCC < 0)

Then compare:
* similarity distributions
* original_score distributions

👉 Hypothesis:
* failure is not random
* likely tied to:
  * high original_score (overfitting)
  * or specific dataset types
  
### 📦 Include additional predictors

Right now you only use:
* similarity

You should include:
* original_score
* dataset type (categorical)
* pipeline complexity (if available)
* variance of similarity distribution

#### Pipeline Complexity:

- number of operators
- pipeline depth (longest path)
- number of unique operators
- total number of parameters (if applicable)
- avg parameter per operator (computed)


👉 Model:

`cross_score ∼ similarity + original_score + interactions`

### >>> 📚 Insight Table

| Pipeline       | Mean Asymetry | Mean Transfer Rate | Mean Transfer Score | # Positive Transfers |
|----------------|----------------|--------------------|---------------------|----------------------|
| Pipeline Label | `T(A → B) - T(B → A)` | `transfer rate= #(cross_score > 0) / total` | `transfer_score = cross_score / original_score` | `%(#(MCC > 0.1))`    |


### >>> 📚 Reuse model

Proposed reuse model:

`reuse proxy ∼ original_score + pipeline complexity + similarity metrics`

Where reuse proxy is one of:

* cross_score
* positive transfer indicator: cross_score > 0
* strong transfer indicator: cross_score > 0.05
* source pipeline mean transfer across targets

And I would strongly consider making the main outcome a binary classification, not only regression.

Why:

* “Will this likely be reusable?” is often more practical than predicting exact MCC
* classification can be more stable when the continuous target is noisy and heavy-tailed

## 💡 >>> Expected Conclusion

__Pipeline reuse is not determined by dataset similarity alone, but by the compatibility between dataset characteristics (e.g., edge density, texture complexity) and the structural composition of the pipeline.__


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