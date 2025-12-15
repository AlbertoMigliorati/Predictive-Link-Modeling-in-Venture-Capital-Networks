# Predictive Link Modeling in Venture Capital Networks

## Research Question

This project addresses the problem of **link prediction** in VC co-investment networks: given a network of past co-investments between VC funds, can we predict which funds will co-invest together in the future?

The approach combines:
- **Graph-based heuristics**: Common Neighbors, Jaccard Coefficient, Preferential Attachment
- **Machine Learning models**: Logistic Regression, Random Forest, Voting Ensemble

## Project Structure

```
Predictive Link Modeling in VC Networks/
├── data/
│   └── raw/
│       └── funding_master.csv      # Investment rounds dataset (2021-2024)
├── results/                        # Output directory for results and plots
├── scr/
│   ├── __init__.py                 # Package initialization
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── models.py                   # Heuristics and ML models
│   └── evaluation.py               # Metrics and visualization
├── main.py                         # Main entry point
├── environment.yml                 # Conda environment specification
├── README.md                       # This file
└── project_report.pdf              # Project report
```

## Dataset

The dataset contains **2,044 investment rounds** from 2021 to 2024, with information about:
- Startup name
- Industries
- Location
- Investor names (comma-separated)
- Lead investor
- Funding type
- Date (Month, Day, Year)


## Installation

### Prerequisites
- Conda (Miniconda or Anaconda)
- Python 3.11+

### Setup

1. Clone or download the project:
```bash
cd Predictive Link Modeling in VC Networks
```

2. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate vc-link-prediction
```

3. Verify installation:
```bash
python -c "import pandas; import networkx; import sklearn; print('All dependencies installed!')"
```

## Usage

### Basic Execution

```bash
python main.py
```


## Methodology

### 1. Data Preprocessing
- Load CSV data with semicolon separator
- Normalize entity names (lowercase, remove suffixes like "Inc", "LLC", "Capital")
- Parse comma-separated investor lists
- Split data by year: train (≤ cutoff_year), test (> cutoff_year)

### 2. Graph Construction
- **Co-investor graph**: Undirected graph where investors are connected if they co-invested in the same startup
- Edge weights represent the number of co-investments

### 3. Link Prediction Heuristics

| Heuristic | Formula | Intuition |
|-----------|---------|-----------|
| Common Neighbors | \|N(u) ∩ N(v)\| | More shared partners → more likely to connect |
| Jaccard Coefficient | \|N(u) ∩ N(v)\| / \|N(u) ∪ N(v)\| | Normalized version of CN |
| Preferential Attachment | \|N(u)\| × \|N(v)\| | Popular nodes attract more connections |

### 4. Machine Learning Models

**Features (5 total):**
- Common Neighbors
- Jaccard Coefficient
- Preferential Attachment
- Degree Sum: deg(u) + deg(v)
- Degree Difference: |deg(u) - deg(v)|

**Models:**
- **Logistic Regression**: Linear model with L2 regularization (C=0.5, balanced class weights)
- **Random Forest**: 200 trees, max depth 15, balanced class weights
- **Voting Ensemble**: Weighted average of heuristics (30%) and ML models (70%)

### 5. Evaluation

**Negative Sampling:**
- Training: Degree-biased sampling for realistic negatives
- Testing: Hard negative sampling (nodes with similar degree to positives)
- Ratio: 5 negatives per positive

**Metrics:**
- **Precision@k**: Fraction of true positives in top-k predictions
- **PR-AUC**: Area under the Precision-Recall curve

## Results

### Performance Summary

| Method | P@50 | P@100 | PR-AUC |
|--------|------|-------|--------|
| Common Neighbors | 1.000 | 0.980 | **0.535** |
| Preferential Attachment | 1.000 | 1.000 | 0.503 |
| Voting Ensemble | 0.860 | 0.780 | 0.463 |
| Logistic Regression | 0.800 | 0.770 | 0.409 |
| Random Forest | 0.620 | 0.690 | 0.395 |
| Jaccard | 0.560 | 0.590 | 0.385 |
| Random Baseline | 0.240 | 0.200 | 0.170 |

### Key Findings

1. **Simple heuristics outperform ML models**: Common Neighbors achieves the highest PR-AUC (0.535), beating both Logistic Regression (0.409) and Random Forest (0.395).

2. **High Precision@100 for top methods**: The top-100 predictions from CN and PA are nearly all correct, indicating reliable identification of likely co-investments.

3. **All methods beat random baseline**: Every method achieves PR-AUC > 0.38, compared to 0.17 for random, demonstrating meaningful predictive power.

4. **Feature importance**: Random Forest relies primarily on Jaccard (51%) and Common Neighbors (37%), confirming that local neighborhood structure is the key predictor.

### Test Edge Difficulty

| Category | Count | Percentage |
|----------|-------|------------|
| Easy (both degree > 10) | 1,407 | 58.8% |
| Medium (both degree 5-10) | 692 | 28.9% |
| Hard (degree < 5) | 293 | 12.2% |
| Very Hard (new nodes) | 0 | 0.0% |

### Sample Top Predictions

1. Alumni Ventures ↔ Tiger Global Management (CN: 34)
2. Andreessen Horowitz ↔ Index Ventures (CN: 34)
3. Craft Ventures ↔ Insight Partners (CN: 34)
4. Felicis Ventures ↔ Y Combinator (CN: 30)
5. Redpoint ↔ Sequoia Capital (CN: 28)

## Output Files

After running `main.py`, the following files are generated in `results/`:

| File | Description |
|------|-------------|
| `precision_at_100.png` | Bar chart of Precision@100 by method |
| `pr_auc.png` | Bar chart of PR-AUC by method |
| `metrics_heatmap.png` | Heatmap of all metrics |
| `top_50_predictions.txt` | Top 50 predicted co-investments |
| `results_coinvestor_*.json` | Full results in JSON format |

## Dependencies

```yaml
- python=3.11
- pandas>=1.5.0
- numpy>=1.23.0
- networkx>=3.0
- scikit-learn>=1.2.0
- matplotlib>=3.6.0
- seaborn>=0.12.0
- pytest>=7.2.0
```

## Limitations

1. **Test edges are between well-connected investors**: 58.8% of test edges involve investors with degree > 10, making prediction easier.

2. **No truly new investors in test set**: All test edges involve investors already present in the training graph.

3. **Limited feature set**: ML models use only 5 features; richer features (industry similarity, geographic proximity) could improve performance.

4. **Temporal dynamics not captured**: The model doesn't account for changing investment patterns over time.
