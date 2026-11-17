# 🛍️ Customer Segmentation

Customer segmentation groups customers into distinct segments based on shared behavior and spending habits, so a business can target each group with the right strategy. This project segments **8,949 credit card customers** with **unsupervised machine learning**. It compares five clustering algorithms, picks the number of clusters with several methods, scores each result with two cluster-quality metrics, and interprets the final segments.

![Python](https://img.shields.io/badge/Language-Python%203.9-3776AB?style=flat-square)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat-square)
![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?style=flat-square)
![Task](https://img.shields.io/badge/Task-Clustering-6A5ACD?style=flat-square)

## 📦 Dataset

👉 [Customer Data on Kaggle](https://www.kaggle.com/datasets/mbsoroush/customer-data) (`Customer_Data.csv`)

**8,950 customers × 18 columns** with no missing values except one `CREDIT_LIMIT` and 313 `MINIMUM_PAYMENTS` entries.

| Group | Features |
|-------|----------|
| **Balance** | `BALANCE`, `BALANCE_FREQUENCY` |
| **Purchases** | `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`, `PURCHASES_TRX` |
| **Cash advances** | `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX` |
| **Payments** | `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT` |
| **Account** | `CREDIT_LIMIT`, `TENURE` |
| **Identifier** | `CUST_ID` (dropped before modeling) |

## 🔄 Workflow

1. **Cleaning:** drop the one row with a missing `CREDIT_LIMIT`, fill the 313 missing `MINIMUM_PAYMENTS` with the median, and drop `CUST_ID` (leaves 8,949 rows × 17 features).
2. **Outlier analysis:** box plots, the IQR rule, and z-scores; an IQR-capped copy of the data is also prepared.
3. **EDA:** feature distributions with mean/median annotations, a pair plot, and a correlation heatmap.
4. **Scaling:** `StandardScaler` on all 17 features.
5. **Choosing *k*:** elbow plot (k = 1 to 20), Silhouette and Calinski-Harabasz scores (k = 2 to 10), and a hybrid score `√(Silhouette × Calinski-Harabasz)`.
6. **Modeling:** K-Means, Hierarchical (Ward), Gaussian Mixture, Mini-Batch K-Means, and DBSCAN (a grid search over `eps` and `min_samples`, guided by a k-distance plot).
7. **Interpretation:** cluster centroids converted back to the original scale and cluster sizes compared across models.
8. **Validation:** classifiers are trained to predict the K-Means cluster from the raw features, to check that the segments are well defined.

## 📊 Results

| Model | Clusters | Silhouette | Calinski-Harabasz | Cluster sizes |
|-------|:--------:|:----------:|:-----------------:|---------------|
| **K-Means** | 3 | **0.25** | **1604.86** | 6,116 / 1,238 / 1,595 |
| Hierarchical (Ward) | 2 | 0.18 | 1418.19 | 5,390 / 3,559 |
| Gaussian Mixture | 2 | 0.20 | 1184.19 | 3,075 / 5,874 |
| Mini-Batch K-Means | 3 | 0.19 | 1517.48 | 5,206 / 502 / 3,241 |
| DBSCAN (`eps=3`, `min_samples=40`) | 1 + noise | 0.58 | 954.36 | 8,626 / 323 noise |
| DBSCAN (`eps=13`, `min_samples=20`) | 1 + noise | 0.85 | 227.47 | 8,943 / 6 noise |

> **Reading the DBSCAN rows:** DBSCAN's higher Silhouette scores come from putting nearly everyone into a single cluster and flagging a few points as noise. That is not a meaningful segmentation, so the scores aren't comparable with the other models. DBSCAN is most useful here for spotting outliers.

**K-Means with k = 3** gave the best balance of cluster separation and usable segments, so it is the main model.

### 🧩 The three K-Means segments

Average values per segment, in the original (unscaled) units:

| | Segment 0 | Segment 1 | Segment 2 |
|---|:---:|:---:|:---:|
| **Customers** | 6,116 | 1,238 | 1,595 |
| Balance | ~800 | ~2,219 | ~3,989 |
| Purchases | ~505 | ~4,266 | ~385 |
| Cash advance | ~331 | ~458 | ~3,866 |
| Purchases frequency | 0.47 | 0.95 | 0.23 |
| Cash-advance frequency | 0.07 | 0.06 | 0.45 |
| Credit limit | ~3,271 | ~7,733 | ~6,675 |
| Full-payment ratio | 0.16 | 0.30 | 0.03 |

- **Segment 0, everyday light users:** low balance, modest and occasional spending, low credit limits.
- **Segment 1, big spenders:** frequent, high-value purchases (both one-off and installment), high credit limits, and the best full-payment ratio.
- **Segment 2, cash-advance users:** high balances driven by large, frequent cash advances, rarely buy, and almost never pay in full.

These segment labels are interpretations of the centroids, so adapt them to your own reading.

### 🌲 Segment validation

Five classifiers were trained to predict the K-Means cluster from the raw features (80/20 split). High accuracy shows the segments are clearly defined in the data:

| Random Forest | Decision Tree | AdaBoost | Gradient Boosting | XGBoost |
|:---:|:---:|:---:|:---:|:---:|
| 0.965 | 0.938 | 0.957 | 0.970 | **0.975** |

## 💡 Conclusion

K-Means and Mini-Batch K-Means produced the most distinct segments (3 clusters), Hierarchical and Gaussian Mixture settled on 2, and DBSCAN mainly isolated outliers in a single dense cluster.

## 📁 Repository Structure

```
CustomerSegmentation/
├── CustomerSegmentation.ipynb   # Full analysis: cleaning, EDA, clustering, evaluation
└── README.md
```

## 🚀 Getting Started

> **Note:** the notebook is large (about 26 MB because of its embedded plots), so it is stored with **Git LFS**. Install [Git LFS](https://git-lfs.com) *before* cloning, otherwise you will get a tiny pointer file instead of the notebook. GitHub's **Download ZIP** button does not include LFS files.

1. Clone the repository:
   ```bash
   git lfs install
   git clone https://github.com/mbsoroush/CustomerSegmentation.git
   cd CustomerSegmentation
   ```
2. Install the dependencies (the notebook was developed with Python 3.9):
   ```bash
   pip install pandas numpy scipy matplotlib seaborn plotly termcolor \
               scikit-learn xgboost kneed yellowbrick jupyter
   ```
   `scikit-learn` 1.2 or newer is needed for the `metric=` argument of `AgglomerativeClustering`.
3. Download `Customer_Data.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/mbsoroush/customer-data).
4. Open the notebook and change the data path in the loading cell to where you saved the file:
   ```python
   data = pd.read_csv('path/to/Customer_Data.csv')
   ```
5. Run the cells in order. Avoid running the pair-plot cell on a slow machine, since it draws all 17 features against each other.

## 🎯 Learning Objectives

- Practice an end-to-end unsupervised learning workflow
- Compare centroid-based, hierarchical, model-based, and density-based clustering
- Choose the number of clusters with elbow, Silhouette, and Calinski-Harabasz methods
- Interpret clusters as actionable customer segments
- Understand the limits of cluster-quality metrics, such as why DBSCAN's scores look high
